"""Shared helpers for the training entrypoints (see finetune_depth.py).

Kept in an importable module (not the __main__ script) so that a checkpoint's
saved config snapshot never depends on the module that produced it.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finetune_depth import FinetuneDepthCfg


def is_rank_zero() -> bool:
    """True on the main process, including before Accelerator/dist init exists
    (torchrun/accelerate launch export RANK / LOCAL_RANK to every process)."""
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def to_primitive(obj):
    """Recursively strip a config to builtin types (enum -> value, dataclass /
    mapping / sequence -> dict / list) so a snapshot of it pickles or serializes
    without needing any project module to reconstruct it."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_primitive(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
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


def resolve_output_dir(cfg: FinetuneDepthCfg, run_hash: str) -> str:
    """Derive the save directory (<save_dir>/<exp_name>_<hash>) and fail fast
    if it already exists: an experiment with this exact config has been run or
    is running, and silently re-running it would waste the compute. To resume
    an interrupted run, pass --resume <path/to/checkpoint-last.pth> explicitly.

    Only rank 0 performs the existence check: under multi-process launch the
    non-zero ranks start later and would otherwise see the directory rank 0
    just created and abort the whole job."""
    output_dir = os.path.join(cfg.save_dir, f"{cfg.exp_name}_{run_hash[:10]}")
    if cfg.resume or not is_rank_zero():
        return output_dir
    if os.path.exists(output_dir):
        raise RuntimeError(
            f"Output dir {output_dir} already exists: an experiment with this exact "
            "config hash has already been launched. Refusing to re-run. Either change "
            f"the config, pass --resume {os.path.join(output_dir, 'checkpoint-last.pth')} "
            "to continue an interrupted run, or remove the directory deliberately."
        )
    return output_dir
