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

import matplotlib
import numpy as np
import torch
import trimesh
from accelerate import Accelerator

from dust3r.inference import loss_of_one_batch  # noqa
from finetune_depth import (
    FinetuneDepthCfg,
    _clip_predictions,
    _prepare_batch,
    _set_data_epoch,
    _stack_depth_batch,
    build_model,
    build_train_loader,
)
from streamvggt.datasets import MultiDatasetConfig, Split
from visual_util import apply_scene_alignment, integrate_camera_into_scene
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


def _per_frame_scene(predictions: dict) -> trimesh.Scene:
    """Build a GLB scene whose per-frame point clouds are SEPARATE, named
    geometries ("frame_000", "frame_001", ...) instead of one fused cloud, so
    the viewer can show/step through frames individually. Cameras for every
    frame are always added (they trace the trajectory). Same world alignment as
    predictions_to_glb (aligned to frame 0), so frames overlay consistently.

    `predictions` is the dict _clip_predictions returns: world_points_from_depth
    [S,H,W,3], depth_conf [S,H,W] (0/1 valid mask here), images [S,3,H,W] in
    [0,1], extrinsic [S,3,4] world->cam."""
    world = predictions["world_points_from_depth"]  # [S,H,W,3]
    conf = predictions["depth_conf"]  # [S,H,W]
    images = predictions["images"]
    extr = predictions["extrinsic"]  # [S,3,4] world->cam
    S = world.shape[0]

    if images.ndim == 4 and images.shape[1] == 3:  # NCHW -> NHWC
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images

    extrinsics_4x4 = np.zeros((S, 4, 4), dtype=np.float32)
    extrinsics_4x4[:, :3, :4] = extr
    extrinsics_4x4[:, 3, 3] = 1.0

    # one global scene scale (from all kept points) so camera markers are sized
    # consistently and the view does not rescale as frames toggle
    kept_all = world.reshape(-1, 3)[conf.reshape(-1) > 1e-5]
    if kept_all.size:
        lo = np.percentile(kept_all, 5, axis=0)
        hi = np.percentile(kept_all, 95, axis=0)
        scene_scale = float(np.linalg.norm(hi - lo)) or 1.0
    else:
        scene_scale = 1.0

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene = trimesh.Scene()
    for i in range(S):
        mask = conf[i].reshape(-1) > 1e-5
        verts = world[i].reshape(-1, 3)[mask]
        cols = (colors[i].reshape(-1, 3)[mask] * 255).astype(np.uint8)
        if verts.size:
            # geom_name -> node name -> three.js object.name (the viewer keys
            # its frame slider off the "frame_NNN" prefix)
            scene.add_geometry(
                trimesh.PointCloud(vertices=verts, colors=cols),
                geom_name=f"frame_{i:03d}",
            )
        rgba = colormap(i / max(S, 1))
        color = tuple(int(255 * x) for x in rgba[:3])
        integrate_camera_into_scene(
            scene, np.linalg.inv(extrinsics_4x4[i]), color, scene_scale
        )

    return apply_scene_alignment(scene, extrinsics_4x4)


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
        "--num-clips", type=int, default=4, help="how many val clips (scenes) to export"
    )
    ap.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="frames per scene; overrides the saved val config (default: as trained, "
        "typically 4). Use e.g. --num-clips 1 --num-views 32 for one long scene.",
    )
    ap.add_argument(
        "--base",
        action="store_true",
        help="visualize the BASE model (pretrained weights in the same architecture, "
        "conditioner/LoRA at init) instead of the finetuned checkpoint. Files are "
        "tagged base_* vs finetuned_* so both can share one --out-dir for A/B.",
    )
    ap.add_argument(
        "--pretrained",
        default=None,
        help="base checkpoint for --base (default: the pretrained path saved in the "
        "run's config; pass explicitly if that relative path does not resolve here).",
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
    if args.num_views is not None:
        # more frames per scene -> longer sequences to scrub. num_views is
        # independent of resolution; the sampler and streaming path handle any
        # S (bounded by sequence length / GPU memory).
        val_ds.num_views = args.num_views

    mode = "base" if args.base else "finetuned"
    out_dir = args.out_dir or str(
        (ckpt_path.parent if ckpt_path.parent != Path("") else Path(".")) / "viz"
    )
    os.makedirs(out_dir, exist_ok=True)

    # A minimal training config: build_model reads pretrained/resume, and
    # build_train_loader reads val_dataset/num_workers/fixed_length. The rest
    # keep their defaults (train_dataset/loss are never touched -- we only
    # build the TEST loader and never run the criterion).
    # --base loads pretrained weights into the same architecture (conditioner /
    # LoRA at their zero-init, so it reproduces the pretrained model's behavior
    # -- the baseline the finetuning improves on).
    pretrained_path = ""
    if args.base:
        pretrained_path = args.pretrained or raw.get("pretrained") or ""
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise SystemExit(
                "--base needs a valid pretrained checkpoint. The path saved in the "
                f"run config was {raw.get('pretrained')!r} (unresolved from here); "
                "pass --pretrained /abs/path/to/base_checkpoint.pth."
            )

    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        val_dataset=val_ds,
        pretrained=pretrained_path,  # only used in --base; else weights load below
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
    # layout matches the saved state_dict.
    if args.base:
        # load_pretrained=True folds the base StreamVGGT weights in; no
        # finetuned state_dict is applied on top.
        print(f"BASE model: loading pretrained weights {pretrained_path}")
        model, _ = build_model(cfg, mcfg, device, load_pretrained=True)
    else:
        model, _ = build_model(cfg, mcfg, device, load_pretrained=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    loader = build_train_loader(cfg, Split.TEST, accelerator, batch_size=1)
    loader = accelerator.prepare(loader)
    _set_data_epoch(loader, 0)  # deterministic clip set/order (see val_loop)

    glb_dir = os.path.join(out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)

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
            views, preds = result["views"], result["pred"]
            confs = _clip_confidences(preds)
            # imgs is not in _stack_depth_batch; stack it the same way
            imgs = torch.stack([v["img"] for v in views], dim=1).float().cpu()
            pred, _, valid, K, pose = _stack_depth_batch(views, preds)
            for b in range(pred.shape[0]):
                if exported >= args.num_clips:
                    break
                predictions = _clip_predictions(
                    imgs[b], pred[b], valid[b], K[b], pose[b]
                )
                scene = _per_frame_scene(predictions)
                scene.export(os.path.join(glb_dir, f"{mode}_clip{exported}.glb"))
                mean_c, min_c, max_c = confs[b]
                print(
                    f"  [{mode}] clip {exported}: {pred.shape[1]} frames | mean depth "
                    f"confidence {mean_c:.4f} (min {min_c:.4f}, max {max_c:.4f})"
                )
                exported += 1
            del result, batch

    print(f"Wrote {exported} .glb file(s) to {glb_dir}")


if __name__ == "__main__":
    main()
