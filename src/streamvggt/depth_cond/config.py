"""Configuration schema for the MeTRIC depth-conditioning module.

Single source of truth: everything downstream (conditioner, injection point,
LoRA wrapping, encoder cache, training wiring) branches on these dataclasses.
Nothing in the module bodies hard-codes the injection point, encoder type,
or temporal setting.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

VALID_ENCODERS = ("identity", "conv", "mae")
VALID_INJECTIONS = ("head", "token")
VALID_TEMPORAL = ("none", "attention")
VALID_NORMS = ("fixed", "raw")
VALID_HEADS = ("depth", "point")
VALID_LORA_TARGETS = ("q", "k", "v", "o")


@dataclass
class DepthCondCfg:
    enabled: bool = True
    # AXIS 1: what encodes the depth
    encoder: str = "identity"  # identity | conv | mae
    # AXIS 2: where depth is injected
    injection: str = "token"  # head (CONTROL) | token (PROPOSED)
    # AXIS 3: temporal mixing inside the conditioner
    temporal: str = "none"  # none | attention

    # depth preprocessing
    space: str = "disparity"
    norm: str = (
        "fixed"  # fixed | raw  (per-frame normalization is FORBIDDEN, see validate())
    )
    norm_constant_m: float = 10.0
    log_depth: bool = True

    # which DPT heads receive head-injection (head arm only)
    heads: List[str] = field(default_factory=lambda: ["depth", "point"])

    # conv encoder width (encoder == "conv")
    conv_channels: int = 128

    # token arm: residual-add into RGB patch tokens (built). Appending extra
    # tokens is a possible future sub-flag; selecting it raises for now.
    token_append: bool = False

    # sparse-depth simulation from GT depthmaps during training
    sim_num_points: int = 512

    def validate(self):
        if self.encoder not in VALID_ENCODERS:
            raise ValueError(
                f"depth_cond.encoder must be one of {VALID_ENCODERS}, got {self.encoder!r}"
            )
        if self.injection not in VALID_INJECTIONS:
            raise ValueError(
                f"depth_cond.injection must be one of {VALID_INJECTIONS}, got {self.injection!r}"
            )
        if self.temporal not in VALID_TEMPORAL:
            raise ValueError(
                f"depth_cond.temporal must be one of {VALID_TEMPORAL}, got {self.temporal!r}"
            )
        if self.space != "disparity":
            raise ValueError("depth_cond.space: only 'disparity' is supported")
        if self.norm not in VALID_NORMS:
            # This is deliberate, not an omission: per-frame / per-sample min-max or
            # max normalization strips absolute metric scale from the input. The model
            # then sees only relative structure and the output silently stops being
            # metric while still looking plausible. Do not add such an option.
            raise ValueError(
                f"depth_cond.norm must be one of {VALID_NORMS}, got {self.norm!r}. "
                "Per-frame normalization is forbidden: it destroys the absolute metric "
                "scale that this whole method exists to inject."
            )
        for h in self.heads:
            if h not in VALID_HEADS:
                raise ValueError(
                    f"depth_cond.heads entries must be in {VALID_HEADS}, got {h!r}"
                )
        if self.token_append:
            raise NotImplementedError(
                "depth_cond.token_append=True (append extra tokens) is not built; "
                "the residual-add path is the default. Leave token_append=False."
            )


@dataclass
class LoRACfg:
    enabled: bool = True
    targets: List[str] = field(default_factory=lambda: ["q", "k", "v", "o"])
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    def validate(self):
        for t in self.targets:
            if t not in VALID_LORA_TARGETS:
                raise ValueError(
                    f"lora.targets entries must be in {VALID_LORA_TARGETS}, got {t!r}"
                )
        if self.enabled and self.rank <= 0:
            raise ValueError("lora.rank must be positive")


@dataclass
class EncoderCacheCfg:
    enabled: bool = False
    dir: str = ""

    def validate(self):
        if self.enabled and not self.dir:
            raise ValueError("encoder_cache.enabled=True requires encoder_cache.dir")


@dataclass
class TrainCondCfg:
    grad_checkpoint: bool = True
    clip_len: Optional[int] = 10  # repo default: num_views=10 in config/finetune.yaml

    def validate(self):
        pass


@dataclass
class MetricCfg:
    depth_cond: DepthCondCfg = field(default_factory=DepthCondCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    encoder_cache: EncoderCacheCfg = field(default_factory=EncoderCacheCfg)
    train: TrainCondCfg = field(default_factory=TrainCondCfg)

    def validate(self):
        self.depth_cond.validate()
        self.lora.validate()
        self.encoder_cache.validate()
        self.train.validate()
        return self


def _build_section(cls, src) -> object:
    """Build a dataclass from a dict-like (plain dict or OmegaConf mapping)."""
    if src is None:
        return cls()
    if not isinstance(src, dict):
        # OmegaConf DictConfig or similar
        src = {str(k): src[k] for k in src}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(src) - names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in src.items():
        if hasattr(v, "_iter_ex") or (
            not isinstance(v, (str, bytes, dict)) and hasattr(v, "__iter__")
        ):
            v = list(v)
        kwargs[k] = v
    return cls(**kwargs)


def build_metric_cfg(cfg) -> MetricCfg:
    """Build a validated MetricCfg from a dict / OmegaConf node with keys
    depth_cond, lora, encoder_cache, train (all optional)."""
    get = (
        (lambda k: cfg.get(k, None))
        if hasattr(cfg, "get")
        else (lambda k: getattr(cfg, k, None))
    )
    mc = MetricCfg(
        depth_cond=_build_section(DepthCondCfg, get("depth_cond")),
        lora=_build_section(LoRACfg, get("lora")),
        encoder_cache=_build_section(EncoderCacheCfg, get("encoder_cache")),
        train=_build_section(TrainCondCfg, get("train")),
    )
    return mc.validate()


# ---------------------------------------------------------------------------
# The confound rule.
#
# The head-vs-token comparison is only interpretable if everything except
# depth_cond.injection is identical between the two runs -- in particular the
# LoRA block. If the token arm trained with different adapters than the head
# arm, a token win could mean "the decoder was allowed to move differently",
# not "caching metric scale helped". These helpers make that a hard failure.
# ---------------------------------------------------------------------------

# Fields excluded from the comparable manifest: exactly the experimental axis
# under study, nothing else.
_CONFOUND_EXEMPT = ("depth_cond.injection",)


def experiment_manifest(cfg: MetricCfg) -> dict:
    """Flat, JSON-serializable manifest of every conditioning/LoRA/cache/train knob."""
    d = dataclasses.asdict(cfg)
    flat = {}

    def _flatten(prefix, obj):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat[key] = list(v) if isinstance(v, (list, tuple)) else v

    _flatten("", d)
    return flat


def manifest_comparable_hash(cfg: MetricCfg) -> str:
    """Hash of the manifest with the exempt axis (injection) removed.
    Two runs are comparable iff their hashes match."""
    flat = experiment_manifest(cfg)
    for k in _CONFOUND_EXEMPT:
        flat.pop(k, None)
    blob = json.dumps(flat, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ConfoundError(RuntimeError):
    pass


def assert_confound_rule(manifest_a: dict, manifest_b: dict):
    """Fail loudly unless the two manifests differ in nothing but injection."""
    keys = set(manifest_a) | set(manifest_b)
    diffs = []
    for k in sorted(keys):
        if k in _CONFOUND_EXEMPT:
            continue
        va, vb = manifest_a.get(k, "<missing>"), manifest_b.get(k, "<missing>")
        if va != vb:
            diffs.append(f"  {k}: {va!r} vs {vb!r}")
    if diffs:
        raise ConfoundError(
            "CONFOUND RULE VIOLATION: runs being compared differ in more than "
            "depth_cond.injection. The head-vs-token comparison is not interpretable.\n"
            + "\n".join(diffs)
        )
