#!/usr/bin/env python
# --------------------------------------------------------
# Run a (base or finetuned) depth-conditioned StreamVGGT on raw SPOT captures
# and export the same artifacts as visualize_depth.py: per-frame point-cloud
# GLBs (serve_glb.py-ready) and optional 2D heatmap series (--heatmaps).
#
# SPOT data (/oscar/data/jtompki1/cli277/new_spot_data/<seq>) has no GT depth,
# no GT poses and no calibrated intrinsics for the raw 640x480 color stream --
# but it has the REAL sparse metric depth (the other stereo camera's point
# cloud projected into this view), which becomes the conditioning input
# directly: simulate_sparse_depth skips views that already carry
# 'sparse_depth', so for the first time the model is conditioned on genuine
# sensor sparsity instead of simulated patch masking.
#
# Cameras: the model predicts its own (pose_enc -> extrinsics + intrinsics).
# To keep a base-vs-finetuned A/B attributable to DEPTH, the pose track is a
# shared reference: the --base run caches its predicted K/pose to
# <out-dir>/pose_cache.npz, and every later run unprojects with the CACHED
# track (its own prediction is only compared against the cache and reported
# as a divergence, which doubles as the "did finetuning move the cameras?"
# diagnostic). Run --base first.
#
# Example (GPU):
#   cd src
#   python visualize_spot.py --weights ../checkpoints/hammer_sweep/b536d87d26e297e1 \
#       --checkpoint best --base --num-views 32 --heatmaps --out-dir ../viz/spot_seq0
#   python visualize_spot.py --weights ../checkpoints/hammer_sweep/b536d87d26e297e1 \
#       --checkpoint best --num-views 32 --heatmaps --out-dir ../viz/spot_seq0
# --------------------------------------------------------
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from dust3r.inference import loss_of_one_batch  # noqa
from finetune_depth import (
    FinetuneDepthCfg,
    _clip_predictions,
    _prepare_batch,
    build_model,
)
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from visualize_depth import (
    _export_heatmaps,
    _per_frame_scene,
    load_saved_args,
    rebuild_metric_cfg,
    resolve_checkpoint,
)

# model input resolutions (multiples of the 14px patch, same as the HAMMER
# training list, which contains both landscape 518x392 and portrait 392x518)
RAW_W, RAW_H = 640, 480
_ROTATIONS = {"none": None, "cw": Image.ROTATE_270, "ccw": Image.ROTATE_90}


def target_dims(rotate: str) -> tuple[int, int]:
    """(width, height) of the model input. SPOT's cameras are mounted sideways,
    so rotating to upright turns the raw landscape frames portrait."""
    return (518, 392) if rotate == "none" else (392, 518)


def read_spot_depth(path: Path) -> np.ndarray:
    """SPOT depth binary: int32 pixel-count header, then HxW float32 metres
    (see new_spot_data/render_depth.py::read_and_process_depth)."""
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if n != RAW_W * RAW_H:
            raise ValueError(f"{path}: header {n}, expected {RAW_W * RAW_H}")
        d = np.fromfile(f, dtype=np.float32, count=n)
    return d.reshape(RAW_H, RAW_W)


def _resize_crop(img: Image.Image, tw: int, th: int, resample) -> Image.Image:
    """Scale to cover (tw, th), then center-crop the overshoot (~5px here).
    Depth/mask must use NEAREST (no blending across holes); color BILINEAR."""
    # TODO: might want floor instead of round here
    s = max(tw / img.width, th / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), resample)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    return img.crop((left, top, left + tw, top + th))


def load_spot_views(
    seq_dir: Path, start: int, num_views: int, stride: int, rotate: str
) -> list[dict]:
    """Build the [S]-list of single-view dicts the streaming path consumes.
    img is ImgNorm-style [-1,1] (matching dataset output; _prepare_batch
    rescales to [0,1]); real sensor sparse depth rides along, so
    simulate_sparse_depth will skip these views."""
    tw, th = target_dims(rotate)
    rot = _ROTATIONS[rotate]
    views = []
    for i in range(num_views):
        idx = start + i * stride
        rgb = Image.open(seq_dir / "color" / f"{idx}.png").convert("RGB")
        if rot is not None:
            rgb = rgb.transpose(rot)
        rgb = _resize_crop(rgb, tw, th, Image.BILINEAR)
        img = torch.from_numpy(np.asarray(rgb).copy()).float().permute(2, 0, 1) / 255.0
        img = img * 2.0 - 1.0  # ImgNorm mean=std=0.5

        depth_im = Image.fromarray(
            read_spot_depth(seq_dir / "depth" / str(idx)), mode="F"
        )
        if rot is not None:
            depth_im = depth_im.transpose(rot)
        depth = np.asarray(_resize_crop(depth_im, tw, th, Image.NEAREST))
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        mask = depth > 0

        views.append(
            {
                "img": img[None],  # [1,3,H,W]
                "sparse_depth": torch.from_numpy(depth.copy())[None],
                "sparse_depth_mask": torch.from_numpy(mask.copy())[None],
                "idx": idx,
                "instance": str(idx),
                "true_shape": torch.tensor([[th, tw]]),
            }
        )
    return views


def predicted_cameras(
    preds: list[dict], hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """pose_enc rows -> (w2c [S,3,4], K [S,3,3]) via the model's own decoder."""
    enc = torch.stack([p["camera_pose"].detach().float().cpu() for p in preds], dim=1)
    extri, intri = pose_encoding_to_extri_intri(enc, hw)
    return extri[0], intri[0]


def to_c2w(w2c: torch.Tensor) -> torch.Tensor:
    """[S,3,4] world->cam  ->  [S,4,4] cam->world."""
    S = w2c.shape[0]
    m = torch.eye(4).repeat(S, 1, 1)
    m[:, :3, :4] = w2c
    return torch.linalg.inv(m)


def pose_divergence(w2c_a: torch.Tensor, w2c_b: torch.Tensor) -> str:
    Ra, Rb = w2c_a[:, :3, :3], w2c_b[:, :3, :3]
    cosang = ((Ra @ Rb.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(1) - 1) / 2
    deg = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
    dt = (w2c_a[:, :3, 3] - w2c_b[:, :3, 3]).norm(dim=1)
    return (
        f"rot mean/max = {deg.mean():.3f}/{deg.max():.3f} deg | "
        f"trans mean/max = {dt.mean():.4f}/{dt.max():.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        required=True,
        help="run dir or .pth (architecture + finetuned weights)",
    )
    ap.add_argument(
        "--checkpoint", choices=["auto", "final", "best", "last"], default="auto"
    )
    ap.add_argument(
        "--base",
        action="store_true",
        help="pretrained weights, conditioning at zero-init; also WRITES the pose cache",
    )
    ap.add_argument(
        "--pretrained",
        default=None,
        help="base checkpoint override (as in visualize_depth)",
    )
    ap.add_argument("--seq-dir", default="/oscar/data/jtompki1/cli277/new_spot_data/0")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--num-views", type=int, default=32)
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="frame stride (SPOT walks fast; try 2-3 if motion is large)",
    )
    ap.add_argument(
        "--rotate",
        choices=["none", "cw", "ccw"],
        default="none",
        help="rotate the raw frames to upright BEFORE the model (SPOT cameras "
        "are mounted sideways). Applied to color and depth alike; the model "
        "then runs at the portrait 392x518 resolution from the training list, "
        "and everything downstream (GLB, heatmaps, poses) is upright.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heatmaps", action="store_true")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = load_saved_args(ckpt)
    mcfg = rebuild_metric_cfg(raw)

    pretrained_path = ""
    if args.base:
        pretrained_path = args.pretrained or raw.get("pretrained") or ""
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise SystemExit(
                f"--base needs a valid pretrained checkpoint (saved: {raw.get('pretrained')!r})"
            )

    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        pretrained=pretrained_path,
        resume=None,
        output_dir=args.out_dir,
    )
    accelerator = Accelerator()
    device = accelerator.device
    if args.base:
        print(f"BASE model: loading pretrained weights {pretrained_path}")
        model, _ = build_model(cfg, mcfg, device, load_pretrained=True)
    else:
        model, _ = build_model(cfg, mcfg, device, load_pretrained=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_dir = Path(args.seq_dir)
    views = load_spot_views(
        seq_dir, args.start, args.num_views, args.stride, args.rotate
    )
    tw, th = target_dims(args.rotate)
    sensor = torch.stack([v["sparse_depth"][0] for v in views])  # [S,H,W], pre-device
    sensor_mask = torch.stack([v["sparse_depth_mask"][0] for v in views])
    dens = sensor_mask.float().mean().item()
    dvals = sensor[sensor_mask]
    print(
        f"sensor sparse depth: {dens:.2%} of pixels | "
        f"range [{dvals.min():.2f}, {dvals.max():.2f}] m, median {dvals.median():.2f}"
    )
    for v in views:
        for k, t in v.items():
            if torch.is_tensor(t):
                v[k] = t.to(device)

    _prepare_batch(views, mcfg)  # rescales img; skips sparse sim (real sparse present)
    with torch.no_grad():
        result = loss_of_one_batch(
            views,
            model,
            None,
            accelerator,
            inference=True,
            symmetrize_batch=False,
            use_amp=True,
        )
    preds = result["pred"]
    pred_depth = torch.stack([p["depth"].detach() for p in preds], dim=1)
    pred_depth = pred_depth.squeeze(-1).float().cpu()[0]  # [S,H,W]
    print(f"pred depth: range [{pred_depth.min():.2f}, {pred_depth.max():.2f}] m")

    # ---- shared pose track (see header) ----
    own_w2c, own_K = predicted_cameras(preds, (th, tw))
    mode = "base" if args.base else "finetuned"
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "pose_cache.npz")
    if args.base:
        np.savez(
            cache_path,
            w2c=own_w2c.numpy(),
            K=own_K.numpy(),
            start=args.start,
            stride=args.stride,
            num_views=args.num_views,
            rotate=args.rotate,
        )
        print(f"pose cache written: {cache_path}")
        w2c, K = own_w2c, own_K
    else:
        if not os.path.exists(cache_path):
            raise SystemExit(
                f"no pose cache at {cache_path}; run --base into this --out-dir first"
            )
        cached = np.load(cache_path)
        for k in ("start", "stride", "num_views"):
            if int(cached[k]) != getattr(args, k):
                raise SystemExit(
                    f"pose cache {k}={int(cached[k])} != {getattr(args, k)}; frames must match"
                )
        if "rotate" in cached and str(cached["rotate"]) != args.rotate:
            raise SystemExit(
                f"pose cache rotate={cached['rotate']} != {args.rotate}; frames must match"
            )
        w2c, K = torch.from_numpy(cached["w2c"]), torch.from_numpy(cached["K"])
        print(f"pose divergence vs cached base track: {pose_divergence(own_w2c, w2c)}")

    c2w = to_c2w(w2c)
    imgs = torch.stack([v["img"][0].float().cpu() for v in views])  # [S,3,H,W] in [0,1]

    # no GT on SPOT: valid = the model's own finite/positive predictions
    predictions = _clip_predictions(
        imgs,
        pred_depth,
        torch.ones_like(pred_depth, dtype=torch.bool),
        K,
        c2w,
        mask_to_gt=False,
    )
    glb_dir = os.path.join(args.out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)
    scene = _per_frame_scene(predictions)
    out_glb = os.path.join(glb_dir, f"{mode}_clip0.glb")
    scene.export(out_glb)
    print(f"wrote {out_glb}")

    if args.heatmaps:
        # 'gterr' here = deviation from the SENSOR's sparse metric depth at its
        # ~valid pixels (dotted maps); tcons is dense and GT-free as always
        n = _export_heatmaps(
            os.path.join(args.out_dir, "heatmaps"),
            f"{mode}_clip0",
            pred_depth,
            sensor,
            sensor_mask,
            K,
            c2w,
        )
        print(f"wrote {n} heatmap PNGs")


if __name__ == "__main__":
    main()
