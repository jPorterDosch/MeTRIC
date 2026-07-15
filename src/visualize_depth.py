#!/usr/bin/env python
# --------------------------------------------------------
# Standalone visualizer for a trained depth-conditioned StreamVGGT run.
#
# Given a weights directory (or a checkpoint file) produced by
# finetune_depth.py, this rebuilds the exact model architecture from the
# checkpoint's saved config, loads the finetuned weights, runs the streaming
# (causal, per-frame KV-cache) inference path on a handful of validation
# clips, and writes each clip's predicted-depth point cloud to a .glb.
#
# It deliberately reuses the training code's helpers -- build_model,
# build_train_loader, _prepare_batch, _set_data_epoch, _export_eval_glbs and
# loss_of_one_batch -- so the visualized geometry is produced by the SAME path
# that eval uses; there is no second, drifting reconstruction of the pipeline.
#
# Requires a CUDA device: loss_of_one_batch runs under torch.cuda.amp.autocast.
#
# Example:
#   cd src
#   python visualize_depth.py --weights ../checkpoints/metric_depth_cond_<id> \
#       --num-clips 4 --checkpoint best
# --------------------------------------------------------
import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator

from dust3r.inference import loss_of_one_batch  # noqa
from finetune_depth import (
    FinetuneDepthCfg,
    _export_eval_glbs,
    _prepare_batch,
    _set_data_epoch,
    build_model,
    build_train_loader,
)
from streamvggt.datasets import MultiDatasetConfig, Split
from streamvggt.depth_cond import (
    DepthCondCfg,
    EncoderCacheCfg,
    LoRACfg,
    MetricCfg,
    TrainCondCfg,
)

# Checkpoint basenames finetune_depth.py writes, best -> worst preference when
# --checkpoint is left at "auto".
_CKPT_NAMES = {
    "final": "checkpoint-final.pth",
    "best": "checkpoint-best.pth",
    "last": "checkpoint-last.pth",
}
_AUTO_ORDER = ("final", "best", "last")


def resolve_checkpoint(weights: str, which: str) -> Path:
    """Resolve --weights (+ --checkpoint) to a concrete .pth. `weights` may be
    the checkpoint file itself or the run directory that contains it."""
    p = Path(weights)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"--weights is neither a file nor a directory: {p}")
    if which != "auto":
        cand = p / _CKPT_NAMES[which]
        if not cand.is_file():
            raise FileNotFoundError(f"No {cand.name} in {p}")
        return cand
    for key in _AUTO_ORDER:
        cand = p / _CKPT_NAMES[key]
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No checkpoint ({', '.join(_CKPT_NAMES.values())}) found in {p}"
    )


def load_saved_args(ckpt: dict) -> dict:
    """The primitive config snapshot picklable_args() embedded at save time.
    Everything is builtin types (enums as strings) -- the config dataclasses'
    validate() coerces them back."""
    if "args" not in ckpt:
        raise KeyError(
            "checkpoint has no 'args' snapshot; cannot reconstruct the model "
            "architecture. Was it written by finetune_depth.py?"
        )
    saved = ckpt["args"]
    return dict(vars(saved)) if not isinstance(saved, dict) else dict(saved)


def rebuild_metric_cfg(raw: dict) -> MetricCfg:
    """Reconstruct the architecture config from the primitive snapshot. Each
    nested dataclass's validate() turns the string/list primitives back into
    enum members, so the built model matches the checkpoint's key layout."""
    return MetricCfg(
        depth_cond=DepthCondCfg(**raw["depth_cond"]),
        lora=LoRACfg(**raw["lora"]),
        encoder_cache=EncoderCacheCfg(**raw["encoder_cache"]),
        train=TrainCondCfg(**raw["train"]),
    ).validate()


def rebuild_val_dataset(raw: dict, data_root: str | None) -> MultiDatasetConfig:
    """Reconstruct the validation dataset config saved with the run. --data-root
    overrides the on-disk location (useful when the data tree lives somewhere
    other than the training CWD); it only supports the single-dataset val
    config finetune_depth.py ships with."""
    vd = dict(raw["val_dataset"])
    if data_root is not None:
        if len(vd["root"]) != 1:
            raise ValueError(
                "--data-root only supports a single-dataset val config; the "
                f"saved config has {len(vd['root'])} roots. Edit the script to "
                "override them individually."
            )
        vd["root"] = [data_root]
        # the lowres loader's highres-exclusion root is a sibling of the old
        # tree; drop it so it does not point outside the overridden root
        if vd.get("highres_root") is not None:
            vd["highres_root"] = [None]
    # The primitive snapshot stores every Path as a plain string, but the
    # dataclass fields are Path-typed and DatasetConfig.validate() calls
    # root.exists(). MultiDatasetConfig.validate() only coerces the enum
    # fields (dataset/split/transform), so convert the path tuples back here
    # or build_all() dies with AttributeError: 'str' has no attribute 'exists'.
    vd["root"] = [Path(r) for r in vd["root"]]
    if vd.get("highres_root") is not None:
        vd["highres_root"] = [
            None if r is None else Path(r) for r in vd["highres_root"]
        ]
    return MultiDatasetConfig(**vd)


def _clip_confidences(preds: list[dict]) -> list[tuple[float, float, float]]:
    """Per-clip (mean, min, max) of the model's predicted depth confidence over
    the whole clip -- a quick read on whether the learned confidences are
    degenerate/saturating (cf. the metric-mode depth_alpha caveat: raw-metre
    residuals can swamp the -alpha*log(sigma) regularizer). preds is the
    [S]-list of per-view dicts; depth_conf is [B,H,W]. Returns one tuple per
    batch element b, in b order (matches _export_eval_glbs's export order)."""
    conf = torch.stack(
        [p["depth_conf"].detach().float() for p in preds], dim=1
    )  # [B,S,H,W]
    flat = conf.reshape(conf.shape[0], -1).cpu()
    return [
        (flat[b].mean().item(), flat[b].min().item(), flat[b].max().item())
        for b in range(flat.shape[0])
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        required=True,
        help="run output dir (containing checkpoint-*.pth) or a .pth file",
    )
    ap.add_argument(
        "--checkpoint",
        choices=["auto", "final", "best", "last"],
        default="auto",
        help="which checkpoint to load from a weights dir (default: auto)",
    )
    ap.add_argument(
        "--num-clips", type=int, default=4, help="how many val clips to export"
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir (default: <weights-dir>/viz). GLBs land in <out-dir>/glb",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="override the saved val dataset root (single-dataset configs only)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "visualize_depth requires a CUDA device (loss_of_one_batch runs "
            "under torch.cuda.amp.autocast)."
        )

    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw = load_saved_args(ckpt)
    mcfg = rebuild_metric_cfg(raw)
    val_ds = rebuild_val_dataset(raw, args.data_root)

    out_dir = args.out_dir or str(
        (ckpt_path.parent if ckpt_path.parent != Path("") else Path(".")) / "viz"
    )
    os.makedirs(out_dir, exist_ok=True)

    # A minimal training config: build_model reads pretrained/resume, and
    # build_train_loader reads val_dataset/num_workers/fixed_length. The rest
    # keep their defaults (train_dataset/loss are never touched -- we only
    # build the TEST loader and never run the criterion).
    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        val_dataset=val_ds,
        pretrained="",  # weights come from the finetuned checkpoint below
        resume=None,
        num_workers=args.num_workers,
        batch_size=1,  # streaming inference needs one clip at a time
        fixed_length=bool(raw.get("fixed_length", True)),
        output_dir=out_dir,
        export_glb=True,
        export_glb_max_clips=args.num_clips,
    )

    accelerator = Accelerator()
    device = accelerator.device

    # build_model applies LoRA + freezes (harmless for eval) so the module key
    # layout matches the saved state_dict; load_pretrained=False because the
    # finetuned weights below supersede the base checkpoint.
    model, _ = build_model(cfg, mcfg, device, load_pretrained=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    loader = build_train_loader(cfg, Split.TEST, accelerator, batch_size=1)
    loader = accelerator.prepare(loader)
    _set_data_epoch(loader, 0)  # deterministic clip set/order (see val_loop)

    exported = 0
    with torch.no_grad():
        for batch in loader:
            if exported >= args.num_clips:
                break
            if batch[0]["img"].shape[0] != 1:
                raise ValueError(
                    "expected a batch-1 loader for streaming inference; got "
                    f"batch size {batch[0]['img'].shape[0]}"
                )
            _prepare_batch(batch, mcfg)
            # inference=True -> the causal per-frame KV-cache path (deployment
            # path), no criterion. result carries views + per-view preds.
            result = loss_of_one_batch(
                batch,
                model,
                None,
                accelerator,
                inference=True,
                symmetrize_batch=False,
                use_amp=True,
            )
            # per-clip confidence before export; confs[j] lines up with the
            # j-th clip _export_eval_glbs writes (both iterate b in order)
            confs = _clip_confidences(result["pred"])
            n = _export_eval_glbs(
                result["views"],
                result["pred"],
                out_dir,
                tag="viz",
                accelerator=accelerator,
                start_idx=exported,
                max_clips=args.num_clips,
            )
            for j in range(n):
                mean_c, min_c, max_c = confs[j]
                print(
                    f"  clip {exported + j}: mean depth confidence {mean_c:.4f} "
                    f"(min {min_c:.4f}, max {max_c:.4f})"
                )
            exported += n
            del result, batch

    print(f"Wrote {exported} .glb file(s) to {os.path.join(out_dir, 'glb')}")


if __name__ == "__main__":
    main()
