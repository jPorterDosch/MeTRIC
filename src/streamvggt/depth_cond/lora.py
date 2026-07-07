"""LoRA adapters for the StreamVGGT aggregator ("decoder") attention.

The repo's Attention uses a single fused nn.Linear(dim, 3*dim) for QKV, so
per-target (q/k/v) adapters are implemented as low-rank updates added to the
corresponding output slice of the fused projection. 'o' wraps attn.proj as a
standard LoRA linear.

Wrapping != unfreezing: the base weights stay frozen (W + B@A with B zero-init,
so the wrapped layer is numerically identical to the base layer at init).
"""

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoRACfg

_QKV_INDEX = {"q": 0, "k": 1, "v": 2}


class _LoRABranch(nn.Module):
    """One low-rank branch: x -> (dropout(x) @ A^T) @ B^T, scaled."""

    def __init__(
        self, in_dim: int, out_dim: int, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        )


class LoRAQKV(nn.Module):
    """Wraps the fused qkv linear; adds independent low-rank updates to the
    q/k/v output slices named in `targets`. Drop-in: forward(x) like nn.Linear."""

    def __init__(
        self,
        base: nn.Linear,
        targets: Iterable[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.dim = base.in_features
        if base.out_features != 3 * self.dim:
            raise ValueError(
                f"expected fused qkv linear (out_features == 3 * in_features), "
                f"got {base.in_features} -> {base.out_features}"
            )
        self.targets = [t for t in targets if t in _QKV_INDEX]
        self.adapters = nn.ModuleDict(
            {
                t: _LoRABranch(self.dim, self.dim, rank, alpha, dropout)
                for t in self.targets
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for t in self.targets:
            i = _QKV_INDEX[t]
            out[..., i * self.dim : (i + 1) * self.dim] += self.adapters[t](x)
        return out


class LoRALinear(nn.Module):
    """Standard LoRA wrapper around a frozen nn.Linear (used for the 'o' proj)."""

    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = _LoRABranch(
            base.in_features, base.out_features, rank, alpha, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)


def apply_lora(aggregator: nn.Module, cfg: LoRACfg) -> int:
    """Wrap the attention projections of every frame and global block.

    Must be called AFTER loading the pretrained checkpoint (wrapping changes
    state-dict key names of the wrapped linears). Returns the number of
    wrapped attention modules.
    """
    if not cfg.enabled:
        return 0
    n = 0
    qkv_targets = [t for t in cfg.targets if t in _QKV_INDEX]
    for blocks in (aggregator.frame_blocks, aggregator.global_blocks):
        for block in blocks:
            attn = block.attn
            if qkv_targets and not isinstance(attn.qkv, LoRAQKV):
                attn.qkv = LoRAQKV(
                    attn.qkv, qkv_targets, cfg.rank, cfg.alpha, cfg.dropout
                )
            if "o" in cfg.targets and not isinstance(attn.proj, LoRALinear):
                attn.proj = LoRALinear(attn.proj, cfg.rank, cfg.alpha, cfg.dropout)
            n += 1
    return n


def lora_parameter_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def param_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
