"""streamvggt multi-view datasets.

A self-contained migration of the DUSt3R dataset machinery (base classes,
sampler, cropping/correspondence/transform utilities) plus the HAMMER,
ARKitScenes and ScanNet loaders, so training no longer imports from the dust3r
tree. See :mod:`streamvggt.datasets.config` for the tyro-exposable
``DatasetConfig`` used to construct these (nested inside a training entrypoint's
config), and :mod:`streamvggt.datasets.types` for the ``Split`` / ``DatasetName``
/ ``TransformName`` enums.
"""

from .arkitscenes import ARKitScenes_Multi
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .base.base_multiview_dataset import EmptyDatasetError
from .base.batched_sampler import BatchedRandomSampler
from .base.easy_dataset import CatDataset
from .config import DatasetConfig, MultiDatasetConfig, build_dataset
from .types import DatasetName, Split, TransformName
from .hammer import HAMMER_Multi
from .hypersim import HyperSim_Multi
from .scannet import ScanNet_Multi

import torch


__all__ = [
    "ARKitScenes_Multi",
    "ARKitScenesHighRes_Multi",
    "HAMMER_Multi",
    "HyperSim_Multi",
    "ScanNet_Multi",
    "BatchedRandomSampler",
    "CatDataset",
    "DatasetConfig",
    "EmptyDatasetError",
    "MultiDatasetConfig",
    "build_dataset",
    "DatasetName",
    "Split",
    "TransformName",
    "get_data_loader",
]


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator=None,
    fixed_length=False,
):
    """Wrap an already-constructed multi-view dataset in a DataLoader driven by
    its aspect-ratio-aware batched sampler.

    The dataset must be a real object (build it with ``DatasetConfig.build()``),
    not a string -- there is no ``eval`` here. A dataset without ``make_sampler``
    raises rather than silently falling back to a plain DataLoader, so a wiring
    mistake surfaces immediately instead of training on the wrong sampler.
    """

    sampler = dataset.make_sampler(
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        world_size=accelerator.num_processes,
        fixed_length=fixed_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
