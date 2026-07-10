"""Smoke test for the migrated streamvggt.datasets package.

Verifies the datasets migrated out of dust3r (HAMMER, ARKitScenes, ScanNet)
are self-contained, fail fast, and satisfy the CUT3R/DUSt3R view contract when
built through the tyro-exposable DatasetConfig.

Checks, in order:
  1. self-containment: nothing under streamvggt.datasets imports the dust3r tree;
  2. registration + exports: the package exposes the datasets, config and
     data-loader factory, and DATASET_REGISTRY covers every DatasetName;
  3. fail-fast: bad max_interval / bad split / missing root raise, not fall back;
  4. tyro CLI: a command line parses into the expected DatasetConfig;
  5. view contract (data-dependent, skipped if the ROOT is absent): every field
     the loss/conditioning code touches exists with the right dtype/shape,
     img in [-1, 1], float32 finite depth, finite cam2world pose, ray_map, ...;
  6. no silent overwrite: max_interval / is_metric passed via the config are the
     values the constructed dataset actually uses.

Run: python tests/streamvggt_datasets_smoke.py [DATA_DIR]
(default DATA_DIR: ~/scratch/data)
"""

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import streamvggt.datasets as D  # noqa: E402
from streamvggt.datasets import DatasetConfig, DatasetName, Split  # noqa: E402
from streamvggt.datasets.config import DATASET_REGISTRY  # noqa: E402

REQUIRED_VIEW_KEYS = [
    "img",
    "depthmap",
    "camera_pose",
    "camera_intrinsics",
    "ray_map",
    "pts3d",
    "valid_mask",
    "true_shape",
    "is_metric",
    "img_mask",
    "ray_mask",
    "dataset",
    "label",
    "instance",
]

# (DatasetName, sub-path under DATA_DIR, canonical max_interval)
DATA_CASES = [
    (DatasetName.HAMMER, "processed_hammer", 20),
    (DatasetName.ARKITSCENES, "processed_arkitscenes", 8),
    (DatasetName.SCANNET, "processed_scannet", 30),
]


def check_self_contained():
    pkg_dir = Path(ROOT) / "src" / "streamvggt" / "datasets"
    offenders = []
    for path in pkg_dir.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import dust3r", "from dust3r")):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {stripped}")
    assert not offenders, "streamvggt.datasets imports dust3r:\n" + "\n".join(offenders)
    print("  [1] self-contained: no dust3r imports")


def check_registration():
    for name in ("HAMMER_Multi", "ARKitScenes_Multi", "ScanNet_Multi",
                 "DatasetConfig", "DatasetName", "Split", "get_data_loader",
                 "build_dataset", "BatchedRandomSampler"):
        assert hasattr(D, name), f"missing export: {name}"
    assert set(DATASET_REGISTRY) == set(DatasetName), "registry misses a DatasetName"
    print("  [2] registration + exports + full registry coverage")


def check_fail_fast():
    root = Path(ROOT)  # exists, so we reach the constructor-level checks
    # bad max_interval
    try:
        DatasetConfig(root=root, dataset=DatasetName.HAMMER, num_views=4,
                      max_interval=0, resolution=(518, 518)).build()
        raise AssertionError("bad max_interval did not raise")
    except ValueError:
        pass
    # missing root
    try:
        DatasetConfig(root=Path("/no/such/root"), dataset=DatasetName.HAMMER,
                      num_views=4, max_interval=20, resolution=(518, 518)).build()
        raise AssertionError("missing root did not raise")
    except FileNotFoundError:
        pass
    # bad split reaches the dataset and is rejected (not silently treated as test)
    try:
        D.ScanNet_Multi(ROOT=str(root), split="trian", num_views=4,
                        resolution=[(518, 518)], max_interval=30)
        raise AssertionError("typo split did not raise")
    except ValueError:
        pass
    print("  [3] fail-fast: bad max_interval / missing root / bad split all raise")


def check_cli():
    import tyro

    argv = [
        "--root", ROOT, "--dataset", "ARKITSCENES", "--num-views", "6",
        "--max-interval", "11", "--resolution", "512", "384",
        "--split", "TEST", "--no-is-metric",
    ]
    cfg = tyro.cli(DatasetConfig, args=argv)
    assert cfg.dataset is DatasetName.ARKITSCENES
    assert cfg.num_views == 6 and cfg.max_interval == 11
    assert cfg.resolution == (512, 384)
    assert cfg.split is Split.TEST and cfg.is_metric is False
    print("  [4] tyro CLI parses into the expected DatasetConfig")


def check_view_contract(data_dir):
    ran = 0
    for dsname, subdir, max_interval in DATA_CASES:
        root = data_dir / subdir
        if not root.exists():
            print(f"  [5] {dsname.name}: SKIP (no data at {root})")
            continue
        cfg = DatasetConfig(root=root, dataset=dsname, num_views=4,
                            max_interval=max_interval, resolution=(518, 518),
                            split=Split.TRAINING, seed=42)
        ds = cfg.build()
        # no silent overwrite
        assert ds.max_interval == max_interval and ds.is_metric is True
        views = ds[0]
        assert len(views) == 4
        for view in views:
            missing = [k for k in REQUIRED_VIEW_KEYS if k not in view]
            assert not missing, f"{dsname.name} view missing {missing}"
            img, depth, valid = view["img"], view["depthmap"], view["valid_mask"]
            assert img.dtype == torch.float32 and img.shape[0] == 3
            assert -1.001 <= float(img.min()) and float(img.max()) <= 1.001
            assert depth.dtype == np.float32 and np.isfinite(depth).all()
            assert np.isfinite(view["camera_pose"]).all()
            assert view["pts3d"].shape[:2] == depth.shape
            assert view["ray_map"].shape == (*depth.shape, 6)
        med = float(np.median(depth[valid])) if valid.any() else -1.0
        print(f"  [5] {dsname.name}: OK  len={len(ds)}  depth_med={med:.2f}m")
        ran += 1
    return ran


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "scratch" / "data"
    print("streamvggt.datasets smoke test")
    check_self_contained()
    check_registration()
    check_fail_fast()
    check_cli()
    ran = check_view_contract(data_dir)
    if ran == 0:
        print("\nPASSED (data-free checks; no dataset ROOTs found to exercise views)")
    else:
        print(f"\nALL CHECKS PASSED ({ran} dataset(s) exercised end to end)")


if __name__ == "__main__":
    main()
