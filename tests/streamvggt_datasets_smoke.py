"""Smoke test for the migrated streamvggt.datasets package.

Verifies the datasets migrated out of dust3r (HAMMER, ARKitScenes, ScanNet)
are self-contained, eval-free, fail fast, and satisfy the CUT3R/DUSt3R view
contract when built through the tyro-exposable DatasetConfig.

Checks, in order:
  1. self-containment: nothing under streamvggt.datasets imports the dust3r tree,
     and the config / package entrypoints contain no eval();
  2. registration + exports: the package exposes the datasets, config, enums and
     data-loader factory, and DatasetName covers exactly the built datasets;
  3. fail-fast: bad max_interval / bad split / missing root raise, not fall back;
  4. enum coercion + tyro CLI: plain strings coerce to enum members, and a
     command line (with member-name choices) parses into the expected config;
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
from streamvggt.datasets import (  # noqa: E402
    DatasetConfig,
    DatasetName,
    Split,
    TransformName,
)

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
    dust3r_offenders = []
    for path in pkg_dir.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import dust3r", "from dust3r")):
                dust3r_offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not dust3r_offenders, "imports dust3r: " + ", ".join(dust3r_offenders)

    # ensure the datasets package stays eval-free (not just entrypoints)
    eval_offenders = []
    for path in pkg_dir.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "eval(" in line:
                eval_offenders.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert not eval_offenders, "eval( found in: " + ", ".join(eval_offenders)
    print("  [1] self-contained: no dust3r imports, no eval() in streamvggt.datasets")


def check_registration():
    for name in (
        "HAMMER_Multi",
        "ARKitScenes_Multi",
        "ScanNet_Multi",
        "DatasetConfig",
        "DatasetName",
        "Split",
        "TransformName",
        "get_data_loader",
        "build_dataset",
        "BatchedRandomSampler",
    ):
        assert hasattr(D, name), f"missing export: {name}"
    assert {d.value for d in DatasetName} == {"hammer", "arkitscenes", "scannet"}
    print("  [2] registration + exports; DatasetName covers the built datasets")


def check_fail_fast():
    root = Path(ROOT)  # exists, so we reach the constructor-level checks
    # bad max_interval (config.validate)
    try:
        DatasetConfig(
            root=root,
            dataset=DatasetName.HAMMER,
            num_views=4,
            max_interval=0,
            resolution=((518, 518),),
        ).build()
        raise AssertionError("bad max_interval did not raise")
    except ValueError:
        pass
    # missing root (config.validate)
    try:
        DatasetConfig(
            root=Path("/no/such/root"),
            dataset=DatasetName.HAMMER,
            num_views=4,
            max_interval=20,
            resolution=((518, 518),),
        ).build()
        raise AssertionError("missing root did not raise")
    except FileNotFoundError:
        pass
    # typo split reaches the dataset match-case fallback and is rejected
    try:
        D.ScanNet_Multi(
            ROOT=str(root),
            split="trian",
            num_views=4,
            resolution=[(518, 518)],
            max_interval=30,
        )
        raise AssertionError("typo split did not raise")
    except ValueError:
        pass
    print("  [3] fail-fast: bad max_interval / missing root / bad split all raise")


def check_enum_and_cli():
    # plain strings coerce to enum members (validate())
    cfg = DatasetConfig(
        root=Path(ROOT),
        dataset="hammer",
        num_views=4,
        max_interval=5,
        resolution=((518, 518),),
        split="train",
        transform="imgnorm",
    )
    cfg.validate()
    assert cfg.dataset is DatasetName.HAMMER and cfg.split is Split.TRAIN
    assert cfg.transform is TransformName.IMGNORM

    # tyro CLI with member-name choices and flattened resolution pairs
    import tyro

    argv = [
        "--root",
        ROOT,
        "--dataset",
        "ARKITSCENES",
        "--num-views",
        "6",
        "--max-interval",
        "11",
        "--resolution",
        "512",
        "384",
        "256",
        "256",
        "--split",
        "TEST",
        "--transform",
        "SEQ_COLOR_JITTER",
        "--no-is-metric",
    ]
    parsed = tyro.cli(DatasetConfig, args=argv)
    assert parsed.dataset is DatasetName.ARKITSCENES
    assert parsed.num_views == 6 and parsed.max_interval == 11
    assert parsed.resolution == ((512, 384), (256, 256))
    assert parsed.split is Split.TEST and parsed.is_metric is False
    assert parsed.transform is TransformName.SEQ_COLOR_JITTER
    print("  [4] enum string coercion + tyro CLI parse into the expected config")


def check_view_contract(data_dir):
    ran = 0
    for dsname, subdir, max_interval in DATA_CASES:
        root = data_dir / subdir
        if not root.exists():
            print(f"  [5] {dsname.name}: SKIP (no data at {root})")
            continue
        cfg = DatasetConfig(
            root=root,
            dataset=dsname,
            num_views=4,
            max_interval=max_interval,
            resolution=((518, 518),),
            split=Split.TRAIN,
            seed=42,
        )
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
    data_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "scratch" / "data"
    )
    print("streamvggt.datasets smoke test")
    check_self_contained()
    check_registration()
    check_fail_fast()
    check_enum_and_cli()
    ran = check_view_contract(data_dir)
    if ran == 0:
        print("\nPASSED (data-free checks; no dataset ROOTs found to exercise views)")
    else:
        print(f"\nALL CHECKS PASSED ({ran} dataset(s) exercised end to end)")


if __name__ == "__main__":
    main()
