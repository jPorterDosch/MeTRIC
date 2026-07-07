"""DepthConditioner: sparse metric depth -> features for a chosen injection point.

One module used by both injection paths. Always operates on [B, S, H, W];
S=1 is the degenerate single-frame case and is never branched on.

Extension point (NOT implemented): a future PoseConditioner (MLP-encoded
rotations/scales, a la MapAnything/POW3R) can be registered as another
conditioner producing the same output contract as `project` here:
  - injection == "head":  {head_name: [per-scale residual [B,S,features,h_i,w_i]]}
  - injection == "token": [B, S, P_patch, token_dim]
Anything meeting that contract can be summed with this module's output at the
injection site in model.py.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DepthCondCfg


def masked_downsample(
    disp: torch.Tensor, mask: torch.Tensor, out_hw, eps: float = 1e-6
):
    """Masked average pooling for sparse maps.

    Standard average pooling is wrong here: it blends "0 m" with "no
    measurement". This pools only over valid pixels and carries the
    fraction-valid map as the new validity channel.

    disp, mask: [N, 1, H, W]; mask in {0, 1}.
    Returns (pooled [N,1,h,w], frac [N,1,h,w]).
    """
    valid_sum = F.adaptive_avg_pool2d(disp * mask, out_hw)  # mean(disp * mask) per cell
    frac = F.adaptive_avg_pool2d(mask, out_hw)  # fraction of valid pixels per cell
    pooled = valid_sum / frac.clamp(min=eps)  # mean over valid pixels only
    pooled = pooled * (frac > 0)  # exactly 0 where a cell had no valid pixel
    return pooled, frac


class ConvDepthEncoder(nn.Module):
    """Small conv stem (2 -> C channels) that brings the sparse (disparity,
    validity) map to the backbone's patch-grid resolution (stride = patch_size),
    following MapAnything's conv depth encoder in spirit."""

    def __init__(self, channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.stem = nn.Conv2d(2, channels, kernel_size=patch_size, stride=patch_size)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, 2, H, W] -> [B, S, C, H/ps, W/ps]
        B, S = x.shape[:2]
        y = self.stem(x.flatten(0, 1))
        y = y + self.refine(y)
        return y.reshape(B, S, *y.shape[1:])


class TemporalAttention(nn.Module):
    """Causal self-attention over the S axis only, applied independently at
    every spatial location. Residual with a zero-initialized output projection,
    so at initialization it is an exact no-op for every S (which subsumes the
    required S=1 no-op)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.num_heads = 1 if dim < 64 else 4
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W] -> same shape
        B, S, C, H, W = x.shape
        t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        h = self.norm(t)
        qkv = self.qkv(h).reshape(B * H * W, S, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [BHW, heads, S, hd]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B * H * W, S, C)
        t = t + self.out(y)
        return t.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)


class DepthConditioner(nn.Module):
    """Sparse metric depth -> features for a chosen injection point.

    Input is always [B, S, H, W] (S=1 = single frame). encoder & temporal both
    optional, selected via cfg (never hard-coded here).

    out_spec:
      - injection == "head":  {"features": int, "num_scales": int, "heads": [names]}
      - injection == "token": {"token_dim": int}
    """

    def __init__(self, cfg: DepthCondCfg, out_spec: dict, patch_size: int = 14):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.injection = cfg.injection
        self.out_spec = out_spec
        self.patch_size = patch_size

        # --- AXIS 1: encoder ---
        if cfg.encoder == "identity":
            self.encoder = None  # raw passthrough ("naive")
            enc_channels = 2
        elif cfg.encoder == "conv":
            self.encoder = ConvDepthEncoder(cfg.conv_channels, patch_size)
            enc_channels = cfg.conv_channels
        elif cfg.encoder == "mae":
            raise NotImplementedError(
                "depth_cond.encoder='mae' (MAE-style encoder for the sparse input) is a "
                "planned variant that has not been built yet. Use 'identity' or 'conv'."
            )
        else:  # pragma: no cover - guarded by cfg.validate()
            raise ValueError(cfg.encoder)
        self._enc_channels = enc_channels

        # --- AXIS 3: temporal mixing (over S only) ---
        self.temporal = (
            TemporalAttention(enc_channels) if cfg.temporal == "attention" else None
        )

        # --- projection to the injection interface ---
        if self.injection == "head":
            features = out_spec["features"]
            num_scales = out_spec["num_scales"]
            # One zero-initialized conv per (target head, fusion scale). By
            # linearity conv([x; d]) == conv(x) + conv(d), so adding these
            # residuals after the head's frozen layer{i}_rn convs is numerically
            # identical to widening those convs' in_channels with zero-init
            # slices -- while leaving the pretrained weights untouched.
            self.head_convs = nn.ModuleDict(
                {
                    head: nn.ModuleList(
                        [
                            nn.Conv2d(
                                enc_channels,
                                features,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            )
                            for _ in range(num_scales)
                        ]
                    )
                    for head in out_spec["heads"]
                }
            )
            for convs in self.head_convs.values():
                for c in convs:
                    nn.init.zeros_(c.weight)
        elif self.injection == "token":
            token_dim = out_spec["token_dim"]
            if cfg.encoder == "identity":
                # Lossless "raw passthrough": fold each patch_size x patch_size
                # patch of both channels into the channel dim, then one linear.
                in_ch = 2 * patch_size * patch_size
            else:
                in_ch = enc_channels
            self.token_proj = nn.Conv2d(in_ch, token_dim, kernel_size=1)
            nn.init.normal_(self.token_proj.weight, std=0.02)
            nn.init.zeros_(self.token_proj.bias)
            # Learnable scalar gate initialized to 0: at init the injected signal
            # is a no-op and the model output equals the pretrained baseline.
            # (Only ONE zero in the path -- a zero gate on top of a zero-init
            # projection would kill both gradients.)
            self.gate = nn.Parameter(torch.zeros(()))
        else:  # pragma: no cover - guarded by cfg.validate()
            raise ValueError(self.injection)

    # ------------------------------------------------------------------
    # Depth preprocessing: where the metric claim lives or dies.
    # ------------------------------------------------------------------
    def prepare(self, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B,S,H,W] depth + mask -> [B,S,2,H,W]: (transformed disparity, validity).

        Works in disparity. norm='fixed' divides disparity by the fixed physical
        constant 1/norm_constant_m (same constant for every frame and sample);
        norm='raw' keeps raw metric disparity. log1p (if enabled) is applied
        AFTER the fixed scaling. All transforms are fixed and monotone, so
        absolute metric scale is preserved. Per-frame normalization is
        forbidden and rejected at config time.

        Invalid pixels carry 0 in channel 0 (not NaN, not a sentinel); the mask
        channel is how the model distinguishes "0 signal" from "no measurement".
        """
        mask = mask.to(dtype=depth.dtype)
        disp = 1.0 / depth.clamp(min=1e-3)
        if self.cfg.norm == "fixed":
            disp = disp * self.cfg.norm_constant_m  # disp / (1 / norm_constant_m)
        if self.cfg.log_depth:
            disp = torch.log1p(disp)
        disp = disp * mask  # a hole and a zero reading must never look the same:
        # holes are 0 in ch0 AND 0 in ch1; readings keep ch1 = 1.
        return torch.stack([disp, mask], dim=2)

    # ------------------------------------------------------------------
    def forward(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        out_hw_list: Optional[List[Tuple[int, int]]] = None,
    ):
        """depth, mask: [B,S,H,W].

        head arm: requires out_hw_list (spatial size of each DPT fusion scale);
                  returns {head_name: [per-scale [B,S,features,h_i,w_i]]}.
        token arm: returns gated token features [B,S,P_patch,token_dim].
        """
        x = self.prepare(depth, mask)  # [B,S,2,H,W]
        if self.encoder is not None:
            x = self.encoder(x)
        if self.temporal is not None:
            x = self.temporal(x)

        if self.injection == "head":
            assert out_hw_list is not None, (
                "head injection needs per-scale output sizes"
            )
            return self._project_head(x, out_hw_list)
        return self._project_token(x)

    def _project_head(
        self, x: torch.Tensor, out_hw_list
    ) -> Dict[str, List[torch.Tensor]]:
        B, S = x.shape[:2]
        flat = x.flatten(0, 1)  # [B*S, C, h, w]
        # Per-scale inputs, once (shared across target heads).
        scale_inputs = []
        for out_hw in out_hw_list:
            if self._enc_channels == 2 and self.encoder is None:
                # identity path: masked multi-scale pooling. Channel 0 is the
                # (possibly temporally-mixed) signal, channel 1 the validity.
                disp = flat[:, 0:1]
                m = flat[:, 1:2]
                pooled, frac = masked_downsample(disp, m, out_hw)
                scale_inputs.append(torch.cat([pooled, frac], dim=1))
            else:
                scale_inputs.append(
                    F.interpolate(
                        flat, size=out_hw, mode="bilinear", align_corners=False
                    )
                )
        out = {}
        for head, convs in self.head_convs.items():
            residuals = []
            for conv, inp in zip(convs, scale_inputs):
                r = conv(inp)
                residuals.append(r.reshape(B, S, *r.shape[1:]))
            out[head] = residuals
        return out

    def _project_token(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        flat = x.flatten(0, 1)  # [B*S, C, h, w]
        if self.encoder is None:
            flat = F.pixel_unshuffle(flat, self.patch_size)  # [B*S, 2*ps^2, ph, pw]
        y = self.token_proj(flat)  # [B*S, token_dim, ph, pw]
        y = y.flatten(2).transpose(1, 2)  # [B*S, P_patch, token_dim]
        y = self.gate * y
        return y.reshape(B, S, *y.shape[1:])


def dpt_fusion_sizes(H: int, W: int, patch_size: int) -> List[Tuple[int, int]]:
    """Spatial size of each DPT fusion scale for an input of size (H, W),
    matching DPTHead.resize_layers: 4x, 2x, 1x, and stride-2 conv (ceil /2)
    of the (H/ps, W/ps) patch grid."""
    ph, pw = H // patch_size, W // patch_size
    return [
        (4 * ph, 4 * pw),
        (2 * ph, 2 * pw),
        (ph, pw),
        ((ph + 1) // 2, (pw + 1) // 2),
    ]
