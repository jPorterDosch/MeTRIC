"""Tyro-exposable configuration for the streamvggt multi-view datasets.

A single :class:`DatasetConfig` fully describes how to construct one dataset
(HAMMER / ARKitScenes / ScanNet). It is meant to be nested inside a training
entrypoint's config (see ``finetune_depth.FinetuneDepthCfg``) so tyro exposes
its fields as ``--dataset.root``, ``--dataset.max-interval`` etc.

Design notes:
  * Fail fast. Every field that genuinely identifies the data (root, dataset,
    num_views, max_interval, resolution) is *required* -- there is no silent
    default that would quietly load the wrong thing. ``validate()`` checks the
    fields up front and each dataset constructor re-checks its own invariants.
  * No silent overwrite. ``is_metric`` and ``max_interval`` are plumbed through
    to the dataset constructors instead of being hardcoded there.
  * No ``eval``. The dataset is selected with a ``match`` over a
    :class:`DatasetName` enum and the transform with a ``match`` over a
    :class:`TransformName` enum -- never by evaluating a string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .arkitscenes import ARKitScenes_Multi
from .base.base_multiview_dataset import BaseMultiViewDataset
from .types import DatasetName, Split, TransformName
from .hammer import HAMMER_Multi
from .scannet import ScanNet_Multi
from .utils.transforms import ColorJitter, ImgNorm, SeqColorJitter


@dataclass
class DatasetConfig:
    """Construct one streamvggt multi-view dataset from CLI-friendly fields.

    Example (Python)::

        cfg = DatasetConfig(
            root=Path("/data/processed_hammer"), dataset=DatasetName.HAMMER,
            num_views=4, max_interval=20, resolution=((518, 518),),
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
    resolution: tuple[tuple[int, int], ...]
    """One or more (width, height) aspect ratios; a batch is sampled at a
    single resolution, and multiple entries enable aspect-ratio augmentation."""

    # --- optional: sensible, explicit defaults ---
    split: Split = Split.TRAIN
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
    transform: TransformName = TransformName.IMGNORM
    """Image transform applied to every view."""
    seed: Optional[int] = None
    """Optional per-sample RNG seed for deterministic sampling."""
    epoch_size: Optional[int] = None
    """If set, resize the dataset to this many samples per epoch (the ``N @``
    operator from EasyDataset); ``None`` leaves it at its natural length."""

    def validate(self) -> "DatasetConfig":
        """Coerce plain strings to enum members and fail fast on any
        inconsistent field or missing data root."""
        # coerce CLI/YAML/test strings to enum members (house style)
        self.dataset = DatasetName(self.dataset)
        self.split = Split(self.split)
        self.transform = TransformName(self.transform)
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if self.max_interval < 1:
            raise ValueError(f"max_interval must be >= 1, got {self.max_interval}")
        if not self.resolution:
            raise ValueError("resolution must list at least one (width, height)")
        for wh in self.resolution:
            if len(wh) != 2 or wh[0] < 1 or wh[1] < 1:
                raise ValueError(
                    f"each resolution must be a positive (width, height), got {wh!r}"
                )
        if self.nneg and self.n_corres <= 0:
            raise ValueError("nneg requires n_corres > 0")
        if self.epoch_size is not None and self.epoch_size < 1:
            raise ValueError(f"epoch_size must be >= 1, got {self.epoch_size}")
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        return self

    def _resolve_transform(self):
        match self.transform:
            case TransformName.IMGNORM:
                return ImgNorm
            case TransformName.SEQ_COLOR_JITTER:
                return SeqColorJitter
            case TransformName.COLOR_JITTER:
                return ColorJitter
            case _:
                raise ValueError(f"Unknown transform: {self.transform!r}")

    def build(self) -> BaseMultiViewDataset:
        """Instantiate and return the configured dataset."""
        self.validate()
        kwargs = dict(
            ROOT=str(self.root),
            split=self.split,
            num_views=self.num_views,
            resolution=[tuple(wh) for wh in self.resolution],
            max_interval=self.max_interval,
            is_metric=self.is_metric,
            aug_crop=self.aug_crop,
            allow_repeat=self.allow_repeat,
            seq_aug_crop=self.seq_aug_crop,
            n_corres=self.n_corres,
            nneg=self.nneg,
            transform=self._resolve_transform(),
            seed=self.seed,
        )
        match self.dataset:
            case DatasetName.HAMMER:
                dataset = HAMMER_Multi(**kwargs)
            case DatasetName.ARKITSCENES:
                dataset = ARKitScenes_Multi(**kwargs)
            case DatasetName.SCANNET:
                dataset = ScanNet_Multi(**kwargs)
            case _:
                raise ValueError(f"Unknown dataset: {self.dataset!r}")

        if self.epoch_size is not None:
            dataset = self.epoch_size @ dataset  # EasyDataset.__rmatmul__
        return dataset


def build_dataset(config: DatasetConfig) -> BaseMultiViewDataset:
    """Functional alias for ``config.build()``."""
    return config.build()
