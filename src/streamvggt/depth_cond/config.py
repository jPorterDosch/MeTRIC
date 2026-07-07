"""Configuration schema for the MeTRIC depth-conditioning module.

Single source of truth: everything downstream (conditioner, injection point,
LoRA wrapping, encoder cache, training wiring) branches on these dataclasses.
Nothing in the module bodies hard-codes the injection point, encoder type,
or temporal setting.
"""

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field


class EncoderType(str, enum.Enum):
    IDENTITY = "identity"  # raw passthrough ("naive")
    CONV = "conv"
    MAE = "mae"  # stub, not built


class InjectionType(str, enum.Enum):
    HEAD = "head"  # into DPT fusion, post-KV-cache (CONTROL)
    TOKEN = "token"  # into encoder tokens, pre-KV-cache (PROPOSED)


class TemporalType(str, enum.Enum):
    NONE = "none"
    ATTENTION = "attention"  # causal self-attention over the S axis


class NormType(str, enum.Enum):
    # Per-frame / per-sample normalization is deliberately NOT representable:
    # it strips absolute metric scale from the input, the model then sees only
    # relative structure, and the output silently stops being metric while
    # still looking plausible. Do not add such a member.
    FIXED = "fixed"
    RAW = "raw"


class HeadType(str, enum.Enum):
    DEPTH = "depth"
    POINT = "point"


class LoRATarget(str, enum.Enum):
    Q = "q"
    K = "k"
    V = "v"
    O = "o"  # noqa: E741 - mirrors the q/k/v/o naming convention


class SparseSimMode(str, enum.Enum):
    NONE = "none"  # no masking: full dense GT depth as conditioning
    RANDOM = "random"  # MAE-style: random visible patches, resampled per frame
    TUBE_MASK = "tube_mask"  # one patch mask shared by every frame of the clip


@dataclass
class DepthCondCfg:
    enabled: bool = True
    # AXIS 1: what encodes the depth
    encoder: EncoderType = EncoderType.IDENTITY
    # AXIS 2: where depth is injected
    injection: InjectionType = InjectionType.TOKEN
    # AXIS 3: temporal mixing inside the conditioner
    temporal: TemporalType = TemporalType.NONE

    # depth preprocessing (always disparity space, see conditioner.prepare)
    norm: NormType = NormType.FIXED
    norm_constant_m: float = 10.0
    log_depth: bool = True

    # which DPT heads receive head-injection (head arm only)
    heads: list[HeadType] = field(
        default_factory=lambda: [HeadType.DEPTH, HeadType.POINT]
    )

    # conv encoder width (encoder == CONV)
    conv_channels: int = 128

    # token arm: residual-add into RGB patch tokens (built). Appending extra
    # tokens is a possible future sub-flag; selecting it raises for now.
    token_append: bool = False

    # sparse-depth simulation from GT depthmaps during training (sparse.py)
    sim_mode: SparseSimMode = SparseSimMode.RANDOM
    sim_patch_size: int = 14
    sim_mask_ratio: float = 0.95  # fraction of patches masked out (invisible)

    def validate(self) -> None:
        # coerce plain strings (CLI / YAML / tests) to enum members
        self.encoder = EncoderType(self.encoder)
        self.injection = InjectionType(self.injection)
        self.temporal = TemporalType(self.temporal)
        self.norm = NormType(self.norm)
        self.heads = [HeadType(h) for h in self.heads]
        self.sim_mode = SparseSimMode(self.sim_mode)
        if self.token_append:
            raise NotImplementedError(
                "depth_cond.token_append=True (append extra tokens) is not built; "
                "the residual-add path is the default. Leave token_append=False."
            )
        if not 0.0 <= self.sim_mask_ratio < 1.0:
            raise ValueError(
                f"sim_mask_ratio must be in [0, 1), got {self.sim_mask_ratio}"
            )
        if self.sim_patch_size <= 0:
            raise ValueError(
                f"sim_patch_size must be positive, got {self.sim_patch_size}"
            )


@dataclass
class LoRACfg:
    enabled: bool = True
    targets: list[LoRATarget] = field(
        default_factory=lambda: [LoRATarget.Q, LoRATarget.K, LoRATarget.V, LoRATarget.O]
    )
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    def validate(self) -> None:
        self.targets = [LoRATarget(t) for t in self.targets]
        # unconditional: a nonsensical rank must never survive into a run,
        # even one that currently has LoRA disabled
        if self.rank <= 0:
            raise ValueError(f"lora.rank must be positive, got {self.rank}")


@dataclass
class EncoderCacheCfg:
    enabled: bool = False
    dir: str = ""

    def validate(self) -> None:
        if self.enabled and not self.dir:
            raise ValueError("encoder_cache.enabled=True requires encoder_cache.dir")


@dataclass
class TrainCondCfg:
    grad_checkpoint: bool = True
    clip_len: int | None = 10  # repo default: num_views=10 in config/finetune.yaml


@dataclass
class MetricCfg:
    depth_cond: DepthCondCfg = field(default_factory=DepthCondCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    encoder_cache: EncoderCacheCfg = field(default_factory=EncoderCacheCfg)
    train: TrainCondCfg = field(default_factory=TrainCondCfg)

    def validate(self) -> "MetricCfg":
        self.depth_cond.validate()
        self.lora.validate()
        self.encoder_cache.validate()
        return self


# ---------------------------------------------------------------------------
# Experiment identity.
#
# manifest = the flattened config itself, minus an explicit blacklist of
# fields that don't change the trained model (save paths, logging cadence).
# hash = SHA-256 over the manifest. The hash names the experiment and its
# save directory; the entrypoint fails fast on a directory collision, so a
# finished experiment is never silently re-run. The manifest is written to
# disk and to wandb, where callers filter runs for comparisons (e.g.
# head-vs-token pairs that agree on every other knob).
# ---------------------------------------------------------------------------


def experiment_manifest(cfg, exclude: tuple[str, ...] = ()) -> dict:
    """Flatten any config dataclass into a JSON-serializable {dotted.key: value}
    dict, dropping the blacklisted top-level field names in `exclude`."""
    d = dataclasses.asdict(cfg)
    for k in exclude:
        d.pop(k, None)
    flat: dict = {}

    def _flatten(prefix: str, obj: dict) -> None:
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                if isinstance(v, (list, tuple)):
                    v = [x.value if isinstance(x, enum.Enum) else x for x in v]
                elif isinstance(v, enum.Enum):
                    v = v.value
                flat[key] = v

    _flatten("", d)
    return flat


def experiment_hash(manifest: dict) -> str:
    """Canonical SHA-256 over a manifest dict. Same config <=> same hash; any
    changed knob (including depth_cond.injection) changes the hash."""
    blob = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
