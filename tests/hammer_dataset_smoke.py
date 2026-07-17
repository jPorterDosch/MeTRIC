"""HAMMER dataset smoke test: the processed data loads through HAMMER_Multi
and satisfies the CUT3R/DUSt3R view contract end to end.

Checks, in order:
  1. registration: the finetune-style eval string builds the dataset;
  2. split integrity: 46 train / 18 test sequences, all metadata consistent;
  3. view contract: every field the loss/conditioning code touches exists
     with the right dtype/shape (img in [-1,1] for the finetune_depth
     (img+1)/2 rescale, float32 depth, finite cam2world pose, ray_map, ...);
  4. metric sanity: depth in a plausible indoor range, mostly valid;
  5. cross-view consistency: pts3d from view A projected into view B's
     camera matches B's own depthmap to a few mm -- this is the check that
     catches a wrong pose convention or intrinsics/crop misalignment, which
     is exactly the silent failure mode we care about.

Run: python tests/hammer_dataset_smoke.py [ROOT]
(default ROOT: ~/scratch/data/processed_hammer)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from streamvggt.datasets import *  # noqa: E402,F403 (the real training loaders)

DATA_ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.expanduser("~/scratch/data/processed_hammer")
)
NUM_VIEWS = 10
RESOLUTION = (518, 392)
# default excludes the *_naked empty-table twins (half of each split); pass
# include_naked=True to load the full 46/18 (checked separately in main).
EXPECTED_SEQS = {"train": 23, "test": 9}
EXPECTED_SEQS_WITH_NAKED = {"train": 46, "test": 18}
VIEW_KEYS = {
    "img",
    "depthmap",
    "camera_pose",
    "camera_intrinsics",
    "dataset",
    "label",
    "instance",
    "is_metric",
    "is_video",
    "quantile",
    "img_mask",
    "ray_mask",
    "camera_only",
    "depth_only",
    "single_view",
    "reset",
    "idx",
    "true_shape",
    "sky_mask",
    "ray_map",
    "pts3d",
    "valid_mask",
    "rng",
}


def build(split):
    # the exact construction path finetune_depth/--train-dataset uses:
    # a string eval'd in the dust3r.datasets namespace
    return eval(
        f"HAMMER_Multi(split='{split}', ROOT='{DATA_ROOT}', "
        f"resolution=[{RESOLUTION}], num_views={NUM_VIEWS}, n_corres=0, "
        f"aug_crop=16, seed=777)"
    )


def check_views(views, tag):
    assert len(views) == NUM_VIEWS, f"{tag}: got {len(views)} views"
    W, H = RESOLUTION
    for v, view in enumerate(views):
        missing = VIEW_KEYS - set(view)
        assert not missing, f"{tag}[{v}]: missing keys {missing}"
        assert view["dataset"] == "hammer"
        assert view["is_metric"] is True

        img = view["img"]
        assert isinstance(img, torch.Tensor) and img.shape == (3, H, W), img.shape
        assert -1.001 <= img.min() and img.max() <= 1.001, (
            f"{tag}[{v}]: img not ImgNorm'd to [-1,1] "
            f"(range {img.min():.3f}..{img.max():.3f})"
        )

        depth = view["depthmap"]
        assert depth.dtype == np.float32 and depth.shape == (H, W)
        assert np.isfinite(depth).all() and depth.min() >= 0
        valid = view["valid_mask"]
        assert valid.dtype == bool and valid.mean() > 0.5, (
            f"{tag}[{v}]: only {valid.mean():.1%} valid depth"
        )
        med = np.median(depth[valid])
        assert 0.1 < med < 5.0, f"{tag}[{v}]: median depth {med:.2f} m implausible"

        pose = view["camera_pose"]
        assert pose.shape == (4, 4) and np.isfinite(pose).all()
        R = pose[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), f"{tag}[{v}]: bad rotation"

        K = view["camera_intrinsics"]
        assert K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0
        assert view["pts3d"].shape == (H, W, 3)
        assert np.isfinite(view["pts3d"][valid]).all()
        assert view["ray_map"].shape == (H, W, 6)


def cross_view_consistency(views, tag, max_med_err_mm=20.0):
    """Project world-frame pts3d of one view into another view's camera and
    compare with that view's own depthmap."""
    checked = 0
    ref = views[0]
    for other in views[1:]:
        X = ref["pts3d"][ref["valid_mask"]]  # world frame (base class applies pose)
        w2c = np.linalg.inv(other["camera_pose"])
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        front = z > 1e-6
        K = other["camera_intrinsics"]
        u = np.round(Xc[front, 0] / z[front] * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(Xc[front, 1] / z[front] * K[1, 1] + K[1, 2]).astype(int)
        H, W = other["depthmap"].shape
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inb.sum() < 1000:
            continue  # views may barely overlap; need enough support to judge
        d_other = other["depthmap"][v[inb], u[inb]]
        m = d_other > 0
        if m.sum() < 1000:
            continue
        err_mm = np.median(np.abs(z[front][inb][m] - d_other[m])) * 1000
        assert err_mm < max_med_err_mm, (
            f"{tag}: median cross-view depth error {err_mm:.1f} mm "
            f"({ref['label']} -> {other['label']}) -- pose/intrinsics misaligned?"
        )
        checked += 1
    assert checked >= 3, f"{tag}: only {checked} view pairs had enough overlap"
    return checked


def main():
    # the include_naked knob: default drops the naked twins, True restores them
    for split in EXPECTED_SEQS:
        n_default = len(build(split).scenes)
        n_with = len(
            eval(
                f"HAMMER_Multi(split='{split}', ROOT='{DATA_ROOT}', "
                f"resolution=[{RESOLUTION}], num_views={NUM_VIEWS}, n_corres=0, "
                f"aug_crop=16, seed=777, include_naked=True)"
            ).scenes
        )
        assert n_default == EXPECTED_SEQS[split], f"{split}: {n_default} (default)"
        assert n_with == EXPECTED_SEQS_WITH_NAKED[split], f"{split}: {n_with} (naked)"
        print(
            f"[{split}] include_naked: {n_default} default / {n_with} with naked -- OK"
        )

    for split, n_expected in EXPECTED_SEQS.items():
        ds = build(split)
        assert len(ds.scenes) == n_expected, (
            f"{split}: {len(ds.scenes)} sequences, expected {n_expected}"
        )
        assert len(ds) > 0 and ds.get_image_num() > 0
        print(
            f"[{split}] {len(ds.scenes)} sequences, {ds.get_image_num()} frames, "
            f"{len(ds)} start ids -- OK"
        )

        # sample a few groups spread across the split
        rng = np.random.default_rng(0)
        for idx in sorted(rng.choice(len(ds), size=3, replace=False)):
            views = ds[(int(idx), 0, NUM_VIEWS)]
            tag = f"{split}/idx{idx}"
            check_views(views, tag)
            n_pairs = cross_view_consistency(views, tag)
            labels = {v["label"].rsplit("_", 1)[0] for v in views}
            assert len(labels) == 1, f"{tag}: views from multiple scenes {labels}"
            print(
                f"  {tag}: contract OK, cross-view consistency OK "
                f"({n_pairs} pairs, scene {labels.pop()})"
            )

    print("\nHAMMER dataset smoke test passed.")


if __name__ == "__main__":
    main()
