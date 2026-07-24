"""Smoke test for the migrated streamvggt.datasets package.

Verifies the datasets migrated out of dust3r (HAMMER, ARKitScenes, ScanNet)
are self-contained, eval-free, fail fast, and satisfy the CUT3R/DUSt3R view
contract when built through the tyro-exposable DatasetConfig.

Checks, in order:
  1. self-containment: nothing under streamvggt.datasets imports the dust3r tree,
     and the config / package entrypoints contain no eval();
  2. registration + exports: the package exposes the datasets, config, enums and
     data-loader factory, and DatasetName covers exactly the built datasets;
  3. fail-fast: bad stride_range / bad split / missing root raise, not fall back;
  4. enum coercion + tyro CLI: plain strings coerce to enum members, and a
     command line (with member-name choices) parses into the expected config;
  5. view contract (data-dependent, skipped if the ROOT is absent): every field
     the loss/conditioning code touches exists with the right dtype/shape,
     img in [-1, 1], float32 finite depth, finite cam2world pose, ray_map, ...;
  6. no silent overwrite: stride_range / is_metric passed via the config are the
     values the constructed dataset actually uses;
  7. multi-dataset config: parallel-tuple fan-out puts entry i of every tuple
     into DatasetConfig i, and any tuple-length mismatch fails fast.

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
    EmptyDatasetError,
    MultiDatasetConfig,
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

# (DatasetName, sub-path under DATA_DIR, canonical stride_range)
DATA_CASES = [
    (DatasetName.HAMMER, "processed_hammer", (1, 20)),
    (DatasetName.ARKITSCENES_LOWRES, "processed_arkitscenes", (1, 8)),
    (DatasetName.ARKITSCENES_HIGHRES, "processed_arkitscenes_highres", (1, 8)),
    (DatasetName.SCANNET, "processed_scannet", (1, 30)),
    (DatasetName.HYPERSIM, "processed_hypersim", (1, 4)),
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
        "HyperSim_Multi",
        "ARKitScenes_Multi",
        "ARKitScenesHighRes_Multi",
        "ScanNet_Multi",
        "CatDataset",
        "DatasetConfig",
        "EmptyDatasetError",
        "MultiDatasetConfig",
        "DatasetName",
        "Split",
        "TransformName",
        "get_data_loader",
        "build_dataset",
        "BatchedRandomSampler",
    ):
        assert hasattr(D, name), f"missing export: {name}"
    assert {d.value for d in DatasetName} == {
        "hammer",
        "arkitscenes_lowres",
        "arkitscenes_highres",
        "scannet",
        "hypersim",
    }
    print("  [2] registration + exports; DatasetName covers the built datasets")


def check_fail_fast():
    root = Path(ROOT)  # exists, so we reach the constructor-level checks
    # bad stride_range (config.validate): lo < 1 and lo > hi
    for bad_stride in ((0, 5), (5, 2)):
        try:
            DatasetConfig(
                root=root,
                dataset=DatasetName.HAMMER,
                num_views=4,
                stride_range=bad_stride,
                resolution=((518, 518),),
            ).build()
            raise AssertionError(f"bad stride_range {bad_stride} did not raise")
        except ValueError:
            pass
    # TEST split demands consecutive frames (stride_range=(1, 1)): temporal
    # metrics assume pixel-aligned adjacency, so a stochastic stride raises
    # both at config validation and at dataset construction
    try:
        DatasetConfig(
            root=root,
            dataset=DatasetName.HAMMER,
            num_views=4,
            stride_range=(1, 20),
            resolution=((518, 518),),
            split=Split.TEST,
        ).build()
        raise AssertionError("TEST + stride_range != (1, 1) did not raise")
    except ValueError:
        pass
    try:
        D.ScanNet_Multi(
            ROOT=str(root),
            split=Split.TEST,
            num_views=4,
            resolution=[(518, 518)],
            stride_range=(1, 30),
        )
        raise AssertionError("constructor TEST + stride != (1, 1) did not raise")
    except ValueError:
        pass
    # missing root (config.validate)
    try:
        DatasetConfig(
            root=Path("/no/such/root"),
            dataset=DatasetName.HAMMER,
            num_views=4,
            stride_range=(1, 20),
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
            stride_range=(1, 30),
        )
        raise AssertionError("typo split did not raise")
    except ValueError:
        pass
    print(
        "  [3] fail-fast: bad stride_range / TEST non-consecutive / "
        "missing root / bad split all raise"
    )


def check_enum_and_cli():
    # plain strings coerce to enum members (validate())
    cfg = DatasetConfig(
        root=Path(ROOT),
        dataset="hammer",
        num_views=4,
        stride_range=(1, 5),
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
        "ARKITSCENES_LOWRES",
        "--num-views",
        "6",
        # parse-only: validate() would demand (1, 1) for the TEST split below;
        # here we only check the pair lands as one (lo, hi) tuple
        "--stride-range",
        "2",
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
    assert parsed.dataset is DatasetName.ARKITSCENES_LOWRES
    assert parsed.num_views == 6 and parsed.stride_range == (2, 11)
    assert parsed.resolution == ((512, 384), (256, 256))
    assert parsed.split is Split.TEST and parsed.is_metric is False
    assert parsed.transform is TransformName.SEQ_COLOR_JITTER
    print("  [4] enum string coercion + tyro CLI parse into the expected config")


def check_multi_config():
    # fan-out: entry i of every parallel tuple lands in DatasetConfig i,
    # shared fields are replicated, and enum strings coerce on validate()
    mc = MultiDatasetConfig(
        root=(Path(ROOT), Path(ROOT)),
        dataset=("arkitscenes_lowres", "arkitscenes_highres"),
        stride_range=((1, 8), (1, 12)),
        epoch_size=(100, 50),
        num_views=4,
        resolution=((518, 518),),
        transform="seq_color_jitter",
    )
    cfgs = mc.to_dataset_configs()
    assert [c.dataset for c in cfgs] == [
        DatasetName.ARKITSCENES_LOWRES,
        DatasetName.ARKITSCENES_HIGHRES,
    ]
    assert [c.stride_range for c in cfgs] == [(1, 8), (1, 12)]
    assert [c.epoch_size for c in cfgs] == [100, 50]
    assert all(c.num_views == 4 and c.is_metric is True for c in cfgs)
    assert all(c.transform is TransformName.SEQ_COLOR_JITTER for c in cfgs)

    # omitted per-dataset optionals: natural length, metric for all
    mc_min = MultiDatasetConfig(
        root=(Path(ROOT),),
        dataset=(DatasetName.HAMMER,),
        stride_range=((1, 20),),
        num_views=4,
        resolution=((518, 518),),
    )
    (cfg,) = mc_min.to_dataset_configs()
    assert cfg.epoch_size is None and cfg.is_metric is True

    # fail-fast: any parallel-tuple length mismatch raises before any build
    for bad in (
        dict(dataset=(DatasetName.HAMMER,)),
        dict(stride_range=((1, 8),)),
        dict(epoch_size=(100,)),
        dict(is_metric=(True,)),
    ):
        kwargs = dict(
            root=(Path(ROOT), Path(ROOT)),
            dataset=(DatasetName.HAMMER, DatasetName.SCANNET),
            stride_range=((1, 20), (1, 30)),
            num_views=4,
            resolution=((518, 518),),
        )
        kwargs.update(bad)
        try:
            MultiDatasetConfig(**kwargs).validate()
            raise AssertionError(f"length mismatch did not raise: {bad}")
        except ValueError:
            pass
    # zero datasets raises
    try:
        MultiDatasetConfig(
            root=(),
            dataset=(),
            stride_range=(),
            num_views=4,
            resolution=((518, 518),),
        ).validate()
        raise AssertionError("empty MultiDatasetConfig did not raise")
    except ValueError:
        pass

    # drift guard: a DatasetConfig knob the fan-out does not pass must raise,
    # not silently take DatasetConfig's default for every dataset
    import dataclasses as dc

    import streamvggt.datasets.config as cfg_mod

    extended = dc.make_dataclass(
        "DatasetConfigWithNewKnob",
        [("new_knob", int, dc.field(default=0))],
        bases=(DatasetConfig,),
    )
    original = cfg_mod.DatasetConfig
    cfg_mod.DatasetConfig = extended
    try:
        mc_min.to_dataset_configs()
        raise AssertionError("unmapped DatasetConfig field did not raise")
    except TypeError as e:
        assert "new_knob" in str(e), f"guard did not name the field: {e}"
    finally:
        cfg_mod.DatasetConfig = original
    print(
        "  [7] multi-dataset config: fan-out correct, length mismatches raise, "
        "unmapped-knob drift guard fires"
    )


def check_view_contract(data_dir):
    ran = 0
    empty_roots = []
    for dsname, subdir, stride_range in DATA_CASES:
        root = data_dir / subdir
        if not root.exists():
            print(f"  [5] {dsname.name}: SKIP (no data at {root})")
            continue
        cfg = DatasetConfig(
            root=root,
            dataset=dsname,
            num_views=4,
            stride_range=stride_range,
            resolution=((518, 518),),
            split=Split.TRAIN,
            seed=42,
        )
        try:
            ds = cfg.build()
        except EmptyDatasetError as e:
            # the root EXISTS but the loader found zero scenes: either a
            # partially-downloaded tree (tolerable) or a scene-glob regression
            # (the exact class this check exists to catch). Tolerate it per
            # dataset but track it -- if NO dataset runs, main() fails rather
            # than reporting a false PASSED.
            empty_roots.append(dsname.name)
            print(f"  [5] {dsname.name}: SKIP ({e})")
            continue
        # no silent overwrite
        assert ds.stride_range == stride_range and ds.is_metric is True
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
    assert ran > 0 or not empty_roots, (
        f"every present data tree produced zero scenes ({empty_roots}): "
        "either the downloads are all stubs or the scene discovery regressed"
    )
    if empty_roots:
        print(
            f"  [5] WARNING: {len(empty_roots)} present tree(s) had zero scenes "
            f"({', '.join(empty_roots)}) -- verify they are partial downloads, "
            "not a loader regression"
        )
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
    check_multi_config()
    ran = check_view_contract(data_dir)
    if ran == 0:
        print("\nPASSED (data-free checks; no dataset ROOTs found to exercise views)")
    else:
        print(f"\nALL CHECKS PASSED ({ran} dataset(s) exercised end to end)")


if __name__ == "__main__":
    main()
