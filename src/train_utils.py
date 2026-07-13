"""Shared helpers for the training entrypoints (see finetune_depth.py).

Kept in an importable module (not the __main__ script) so that a checkpoint's
saved config snapshot never depends on the module that produced it. Named
`train_utils` rather than `utils` deliberately: a top-level `utils` module
shadows the vendored `src/croco/utils` namespace package once `src/` is on the
path, breaking croco's own `import utils.misc`.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finetune_depth import FinetuneDepthCfg
    from streamvggt.loss import LossConfig

# Rank env vars exported by the launchers we support, in priority order.
# accelerate/torchrun set RANK/LOCAL_RANK; SLURM srun sets SLURM_PROCID; MPI
# launchers set OMPI_COMM_WORLD_RANK / PMI_RANK. The first one present wins.
_RANK_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
)


def is_rank_zero() -> bool:
    """True on the main process, including before Accelerator/dist init exists.
    Checks several launcher conventions so a multi-GPU srun/MPI job (which does
    not export RANK the way torchrun does) still identifies its non-zero ranks
    -- otherwise every rank would run the resolve_output_dir existence check and
    the later ranks would abort on the directory rank 0 just created."""
    for var in _RANK_ENV_VARS:
        if var in os.environ:
            return os.environ[var] == "0"
    return True


def loss_from_cfg(node) -> "LossConfig":
    """Build a LossConfig from a hydra/OmegaConf node.

    The hydra entrypoints load config as an OmegaConf DictConfig of plain
    scalars (string-valued enums); this converts it to a plain dict and hands it
    to LossConfig, whose __post_init__ coerces the strings back to enums. The
    tyro entrypoint (finetune_depth) skips all this -- tyro already produces a
    typed LossConfig, so it just calls ``args.loss.build()`` directly.

    Keeping this OmegaConf glue here rather than in the loss package leaves
    streamvggt.loss.types framework-agnostic. Imports are local so importing
    train_utils (e.g. for the checkpoint-config path) stays cheap and does not
    pull in torch or OmegaConf.
    """
    from omegaconf import OmegaConf
    from streamvggt.loss import LossConfig

    return LossConfig(**OmegaConf.to_container(node, resolve=True))


def to_primitive(obj):
    """Recursively strip a config to builtin types (enum -> value, dataclass /
    mapping / sequence -> dict / list) so a snapshot of it pickles or serializes
    without needing any project module to reconstruct it."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_primitive(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, dict):
        return {k: to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]
    return obj


def picklable_args(cfg: FinetuneDepthCfg) -> argparse.Namespace:
    """Config snapshot safe to embed in checkpoints. Two hazards are avoided:
    (1) FinetuneDepthCfg lives in __main__, so pickling the dataclass makes the
    checkpoint unloadable from any other script; (2) the nested config holds
    str-Enum members whose pickle still records their class, so unpickling would
    require the streamvggt package on the path. to_primitive reduces everything
    to builtins, and the Namespace still offers the attribute access croco's
    misc.save_model needs (args.output_dir)."""
    return argparse.Namespace(**to_primitive(cfg))


def resolve_output_dir(cfg: FinetuneDepthCfg, run_id: str) -> str:
    """Derive the save directory for this run. `run_id` is the already-truncated
    short experiment id (see config.experiment_id) -- it is used verbatim, never
    re-sliced here, so the truncation length lives in exactly one place.

    Resume: continue the run that OWNS the checkpoint -- the output dir is the
    checkpoint's parent -- regardless of any identity-knob drift in the current
    config (e.g. a bumped --epochs to extend the run). Deriving it from the
    current id instead would silently fork the resumed run into a fresh
    directory, splitting one logical run across two dirs / wandb runs.

    Fresh run: <save_dir>/<exp_name>_<run_id>, failing fast if it already exists
    (an experiment with this exact config has been run or is running, and
    silently re-running it would waste the compute). Only rank 0 performs the
    existence check: under multi-process launch the non-zero ranks start later
    and would otherwise see the directory rank 0 just created and abort the job.
    """
    if cfg.resume:
        return os.path.dirname(os.path.abspath(cfg.resume))
    output_dir = os.path.join(cfg.save_dir, f"{cfg.exp_name}_{run_id}")
    if not is_rank_zero():
        return output_dir
    if os.path.exists(output_dir):
        raise RuntimeError(
            f"Output dir {output_dir} already exists: an experiment with this exact "
            "config hash has already been launched. Refusing to re-run. Either change "
            f"the config, pass --resume {os.path.join(output_dir, 'checkpoint-last.pth')} "
            "to continue an interrupted run, or remove the directory deliberately."
        )
    return output_dir
