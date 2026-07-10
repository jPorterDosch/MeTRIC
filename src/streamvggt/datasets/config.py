"""Tyro-exposable configuration for the streamvggt multi-view datasets.

A single :class:`DatasetConfig` fully describes how to construct one dataset
(HAMMER / ARKitScenes / ScanNet). It is a plain stdlib dataclass so it can be
imported without tyro installed; ``tyro`` is only imported lazily inside
:func:`cli`. Build a dataset object with ``DatasetConfig(...).build()``.

Design notes:
  * Fail fast. Every field that genuinely identifies the data (root, dataset,
    num_views, max_interval, resolution) is *required* -- there is no silent
    default that would quietly load the wrong thing. Validation happens up
    front in ``build()`` and again inside each dataset constructor.
  * No silent overwrite. ``is_metric`` and ``max_interval`` are plumbed through
    to the dataset constructors instead of being hardcoded there, so a value
    you pass is the value that is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from streamvggt.datasets.arkitscenes import ARKitScenes_Multi
from streamvggt.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from streamvggt.datasets.hammer import HAMMER_Multi
from streamvggt.datasets.scannet import ScanNet_Multi


class Split(Enum):
    """Train/test split. The value is the canonical string each dataset
    constructor expects."""

    TRAINING = "train"
    TEST = "test"


class DatasetName(Enum):
    """Which dataset a :class:`DatasetConfig` builds."""

    HAMMER = "hammer"
    ARKITSCENES = "arkitscenes"
    SCANNET = "scannet"


# Single source of truth mapping the CLI enum to the concrete dataset class.
DATASET_REGISTRY: dict[DatasetName, type[BaseMultiViewDataset]] = {
    DatasetName.HAMMER: HAMMER_Multi,
    DatasetName.ARKITSCENES: ARKitScenes_Multi,
    DatasetName.SCANNET: ScanNet_Multi,
}


@dataclass
class DatasetConfig:
    """Construct one streamvggt multi-view dataset from CLI-friendly fields.

    Example (tyro CLI)::

        python -m streamvggt.datasets.config \\
            --root /data/processed_hammer --dataset HAMMER \\
            --num-views 4 --max-interval 20 --resolution 518 518 \\
            --split TRAINING

    Example (Python)::

        cfg = DatasetConfig(
            root=Path("/data/processed_hammer"), dataset=DatasetName.HAMMER,
            num_views=4, max_interval=20, resolution=(518, 518),
        )
        dataset = cfg.build()
    """

    # --- required: uniquely identify the data to load (no silent defaults) ---
    root: Path
    """Filesystem root of the preprocessed dataset."""
    dataset: DatasetName
    """Which dataset to build."""
    num_views: int
    """Number of views per sample."""
    max_interval: int
    """Maximum frame interval when sampling a view sequence."""
    resolution: tuple[int, int]
    """Target (width, height) the images/depthmaps are cropped and resized to."""

    # --- optional: sensible, explicit defaults ---
    split: Split = Split.TRAINING
    """Train or test split."""
    is_metric: bool = True
    """Whether the depth/pose are in metric scale."""
    aug_crop: int = 0
    """Random crop augmentation budget in pixels (0 disables it)."""
    allow_repeat: bool = False
    """Allow repeating frames to reach ``num_views`` in short sequences."""
    seq_aug_crop: bool = False
    """Use one shared crop delta across a sampled sequence."""
    n_corres: int = 0
    """Number of correspondences to extract per view (0 disables it)."""
    nneg: int = 0
    """Number of negative correspondences (only valid when ``n_corres`` > 0)."""
    transform: str = "ImgNorm"
    """Image transform name, resolved in the dataset transforms namespace
    (e.g. ``ImgNorm`` or ``SeqColorJitter``)."""
    seed: Optional[int] = None
    """Optional per-sample RNG seed for deterministic sampling."""

    def _validate(self) -> None:
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if self.max_interval < 1:
            raise ValueError(f"max_interval must be >= 1, got {self.max_interval}")
        w, h = self.resolution
        if w < 1 or h < 1:
            raise ValueError(
                f"resolution must be positive (width, height), got {self.resolution}"
            )
        if self.nneg and self.n_corres <= 0:
            raise ValueError("nneg requires n_corres > 0")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

    def build(self) -> BaseMultiViewDataset:
        """Instantiate and return the configured dataset (fails fast on any
        inconsistent field or missing data root)."""
        self._validate()
        dataset_cls = DATASET_REGISTRY[self.dataset]
        return dataset_cls(
            ROOT=str(self.root),
            split=self.split.value,
            num_views=self.num_views,
            resolution=tuple(self.resolution),
            max_interval=self.max_interval,
            is_metric=self.is_metric,
            aug_crop=self.aug_crop,
            allow_repeat=self.allow_repeat,
            seq_aug_crop=self.seq_aug_crop,
            n_corres=self.n_corres,
            nneg=self.nneg,
            transform=self.transform,
            seed=self.seed,
        )


def build_dataset(config: DatasetConfig) -> BaseMultiViewDataset:
    """Functional alias for ``config.build()``."""
    return config.build()


def cli() -> DatasetConfig:
    """Parse a :class:`DatasetConfig` from the command line via tyro."""
    import tyro

    return tyro.cli(DatasetConfig)


if __name__ == "__main__":
    cfg = cli()
    dataset = cfg.build()
    print(dataset)
    print(dataset.get_stats())
