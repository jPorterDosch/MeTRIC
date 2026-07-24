"""Tyro-exposable configuration for the streamvggt multi-view datasets.

A single :class:`DatasetConfig` fully describes how to construct one dataset
(HAMMER / ARKitScenes lowres / ARKitScenes highres / ScanNet), and
:class:`MultiDatasetConfig` describes N of them as parallel per-dataset tuples
(combine the built datasets with ``+`` in the entrypoint). DatasetConfig is
meant to be nested inside a training
entrypoint's config (see ``finetune_depth.FinetuneDepthCfg``) so tyro exposes
its fields as ``--dataset.root``, ``--dataset.stride-range`` etc.

Design notes:
  * Fail fast. Every field that genuinely identifies the data (root, dataset,
    num_views, stride_range, resolution) is *required* -- there is no silent
    default that would quietly load the wrong thing. ``validate()`` checks the
    fields up front and each dataset constructor re-checks its own invariants.
  * No silent overwrite. ``is_metric`` and ``stride_range`` are plumbed through
    to the dataset constructors instead of being hardcoded there.
  * No ``eval``. The dataset is selected with a ``match`` over a
    :class:`DatasetName` enum and the transform with a ``match`` over a
    :class:`TransformName` enum -- never by evaluating a string.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from .arkitscenes import ARKitScenes_Multi
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .base.base_multiview_dataset import BaseMultiViewDataset, validate_stride_range
from .types import DatasetName, Split, TransformName
from .hammer import HAMMER_Multi
from .hypersim import HyperSim_Multi
from .scannet import ScanNet_Multi
from .utils.transforms import ColorJitter, ImgNorm, SeqColorJitter


@dataclass
class DatasetConfig:
    """Construct one streamvggt multi-view dataset from CLI-friendly fields.

    Example (Python)::

        cfg = DatasetConfig(
            root=Path("/data/processed_hammer"), dataset=DatasetName.HAMMER,
            num_views=4, stride_range=(1, 20), resolution=((518, 518),),
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
    stride_range: tuple[int, int]
    """(lo, hi) frame-stride bounds: one stride is drawn uniformly in
    [lo, hi] per clip. ``(1, 1)`` = consecutive frames -- required for the
    TEST split, where temporal metrics assume pixel-aligned adjacency."""
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
    regular_stride: bool = True
    """``True``: one stride per clip, i.e. a constant frame rate. ``False``:
    gaps drawn independently per adjacent pair, so the rate varies within the
    clip. An ablation axis, hence a config choice rather than a per-sample coin
    flip; a no-op when ``stride_range`` has ``lo == hi``."""
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
    highres_root: Optional[Path] = None
    """ARKITSCENES_LOWRES only: explicit root of the highres sibling tree whose
    scenes the lowres loader excludes (fails fast if missing). ``None`` falls
    back to the original DUSt3R convention of deriving ``ROOT + "_highres"``
    and silently skipping exclusion when that tree is absent."""
    include_naked: bool = False
    """HAMMER only: keep the ``*_naked`` sequences (each object scene's
    empty-table recapture along the same trajectory -- near-duplicate, trivial
    geometry). Default ``False`` drops them (23/46 train, 9/18 test). A no-op
    for the other datasets, which have no naked twins."""

    def validate(self) -> "DatasetConfig":
        """Coerce plain strings to enum members and fail fast on any
        inconsistent field or missing data root."""
        # coerce CLI/YAML/test strings to enum members (house style)
        self.dataset = DatasetName(self.dataset)
        self.split = Split(self.split)
        self.transform = TransformName(self.transform)
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        # one shared guard (see base_multiview_dataset): a scalar left over from
        # the max_interval spelling, a reversed pair, or a numpy int from a
        # sweep all fail here with the same message the constructors give
        self.stride_range = validate_stride_range(self.stride_range, "DatasetConfig")
        if self.split is Split.TEST and self.stride_range != (1, 1):
            raise ValueError(
                f"TEST split requires stride_range=(1, 1) (consecutive frames; "
                f"temporal metrics assume pixel-aligned adjacency), "
                f"got {self.stride_range!r}"
            )
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
        if (
            self.highres_root is not None
            and self.dataset is not DatasetName.ARKITSCENES_LOWRES
        ):
            raise ValueError(
                f"highres_root only applies to {DatasetName.ARKITSCENES_LOWRES}, "
                f"got dataset={self.dataset}"
            )
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
            stride_range=self.stride_range,  # normalized by validate() above
            regular_stride=self.regular_stride,
            is_metric=self.is_metric,
            aug_crop=self.aug_crop,
            seq_aug_crop=self.seq_aug_crop,
            n_corres=self.n_corres,
            nneg=self.nneg,
            transform=self._resolve_transform(),
            seed=self.seed,
        )
        match self.dataset:
            case DatasetName.HAMMER:
                dataset = HAMMER_Multi(include_naked=self.include_naked, **kwargs)
            case DatasetName.ARKITSCENES_LOWRES:
                dataset = ARKitScenes_Multi(
                    highres_root=(
                        None if self.highres_root is None else str(self.highres_root)
                    ),
                    **kwargs,
                )
            case DatasetName.ARKITSCENES_HIGHRES:
                dataset = ARKitScenesHighRes_Multi(**kwargs)
            case DatasetName.SCANNET:
                dataset = ScanNet_Multi(**kwargs)
            case DatasetName.HYPERSIM:
                dataset = HyperSim_Multi(**kwargs)
            case _:
                raise ValueError(f"Unknown dataset: {self.dataset!r}")

        if self.epoch_size is not None:
            dataset = self.epoch_size @ dataset  # EasyDataset.__rmatmul__
        return dataset


def build_dataset(config: DatasetConfig) -> BaseMultiViewDataset:
    """Functional alias for ``config.build()``."""
    return config.build()


@dataclass
class MultiDatasetConfig:
    """Construct N streamvggt datasets from parallel per-dataset tuples.

    Per-dataset fields (``root``, ``dataset``, ``stride_range``, and optionally
    ``epoch_size`` / ``is_metric``) are ordered tuples indexed together: entry
    ``i`` of every tuple describes dataset ``i``. All provided tuples must have
    the same length -- ``validate()`` fails fast on any mismatch. The remaining
    fields are shared by every dataset; ``num_views`` and ``resolution`` in
    particular *must* be shared because ``CatDataset`` requires them to agree
    across concatenated datasets.

    ``build_all()`` returns the datasets in order (each already resized by its
    ``epoch_size`` via the ``N @`` operator). Combining them -- e.g. summing
    with ``+`` into a ``CatDataset`` -- is the caller's job, so the mixture
    lives in the training entrypoint, not here.

    Example (CLI)::

        --dataset.root /data/lowres /data/highres \\
        --dataset.dataset arkitscenes_lowres arkitscenes_highres \\
        --dataset.stride-range 1 8 1 8 --dataset.epoch-size 4500 2250
    """

    # --- per-dataset parallel tuples (equal length, fail fast) ---
    root: tuple[Path, ...]
    """Filesystem root of each preprocessed dataset."""
    dataset: tuple[DatasetName, ...]
    """Which dataset to build at each root."""
    stride_range: tuple[tuple[int, int], ...]
    """Per-dataset (lo, hi) frame-stride bounds (one stride drawn uniformly in
    [lo, hi] per clip; ``(1, 1)`` = consecutive, required for TEST)."""

    # --- shared: identical for every dataset ---
    num_views: int
    """Number of views per sample (must match across concatenated datasets)."""
    resolution: tuple[tuple[int, int], ...]
    """Shared (width, height) aspect-ratio list (must match across datasets)."""

    # --- per-dataset, optional ---
    epoch_size: Optional[tuple[int, ...]] = None
    """Per-dataset samples per epoch (the ``N @`` weights). Either one entry
    per dataset or omitted entirely (natural lengths)."""
    is_metric: Optional[tuple[bool, ...]] = None
    """Per-dataset metric-scale flags; omitted means metric for all."""
    highres_root: Optional[tuple[Optional[Path], ...]] = None
    """Per-dataset explicit highres exclusion root (see
    ``DatasetConfig.highres_root``); only meaningful for ARKITSCENES_LOWRES
    entries -- use ``None`` for the others. Omitted means the DUSt3R naming
    convention for every dataset."""

    # --- shared, optional ---
    split: Split = Split.TRAIN
    aug_crop: int = 0
    regular_stride: bool = True
    """Shared across the mixture (see ``DatasetConfig.regular_stride``): the
    regular/irregular choice is a property of the training objective, not of an
    individual dataset's capture rate -- that is what ``stride_range`` is for,
    which is why only that one is per-dataset."""
    seq_aug_crop: bool = False
    n_corres: int = 0
    nneg: int = 0
    transform: TransformName = TransformName.IMGNORM
    seed: Optional[int] = None
    include_naked: bool = False
    """HAMMER only: keep the ``*_naked`` empty-table sequences (default ``False``
    drops them). Shared across the mixture; a no-op for datasets without naked
    twins."""

    def validate(self) -> "MultiDatasetConfig":
        """Fail fast on parallel-tuple length mismatches; per-dataset field
        validation is delegated to each ``DatasetConfig.validate()``."""
        # coerce CLI/YAML/test strings to enum members (house style)
        self.dataset = tuple(DatasetName(d) for d in self.dataset)
        self.split = Split(self.split)
        self.transform = TransformName(self.transform)
        n = len(self.root)
        if n < 1:
            raise ValueError("MultiDatasetConfig needs at least one dataset root")
        for name, values in (
            ("dataset", self.dataset),
            ("stride_range", self.stride_range),
            ("epoch_size", self.epoch_size),
            ("is_metric", self.is_metric),
            ("highres_root", self.highres_root),
        ):
            if values is not None and len(values) != n:
                raise ValueError(
                    f"MultiDatasetConfig length mismatch: {n} roots but "
                    f"{len(values)} {name} entries"
                )
        return self

    def to_dataset_configs(self) -> list[DatasetConfig]:
        """Fan the parallel tuples out into one ``DatasetConfig`` per dataset."""
        self.validate()
        configs = []
        for i in range(len(self.root)):
            kwargs = dict(
                root=self.root[i],
                dataset=self.dataset[i],
                num_views=self.num_views,
                stride_range=self.stride_range[i],
                resolution=self.resolution,
                split=self.split,
                is_metric=True if self.is_metric is None else self.is_metric[i],
                aug_crop=self.aug_crop,
                regular_stride=self.regular_stride,
                seq_aug_crop=self.seq_aug_crop,
                n_corres=self.n_corres,
                nneg=self.nneg,
                transform=self.transform,
                seed=self.seed,
                epoch_size=None if self.epoch_size is None else self.epoch_size[i],
                highres_root=(
                    None if self.highres_root is None else self.highres_root[i]
                ),
                include_naked=self.include_naked,
            )
            if i == 0:
                # a DatasetConfig knob this fan-out does not pass would
                # silently take DatasetConfig's default for every dataset;
                # fail loudly instead so the omission is a wiring error, not
                # a quietly-ignored setting
                unmapped = {f.name for f in fields(DatasetConfig)} - set(kwargs)
                if unmapped:
                    raise TypeError(
                        f"DatasetConfig field(s) not exposed by "
                        f"MultiDatasetConfig: {sorted(unmapped)}; add each as a "
                        f"per-dataset tuple or shared field and pass it in "
                        f"to_dataset_configs()"
                    )
            configs.append(DatasetConfig(**kwargs))
        return configs

    def build_all(self) -> list[BaseMultiViewDataset]:
        """Instantiate every configured dataset, in order."""
        return [config.build() for config in self.to_dataset_configs()]
