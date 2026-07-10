"""streamvggt multi-view datasets.

A self-contained migration of the DUSt3R dataset machinery (base classes,
sampler, cropping/correspondence/transform utilities) plus the HAMMER,
ARKitScenes and ScanNet loaders, so training no longer imports from the dust3r
tree. See :mod:`streamvggt.datasets.config` for the tyro-exposable
``DatasetConfig`` used to construct these from the CLI.
"""

from accelerate import Accelerator

from streamvggt.datasets.base.batched_sampler import BatchedRandomSampler  # noqa: F401
from streamvggt.datasets.utils.transforms import *  # noqa: F401,F403

from streamvggt.datasets.arkitscenes import ARKitScenes_Multi  # noqa: F401
from streamvggt.datasets.hammer import HAMMER_Multi  # noqa: F401
from streamvggt.datasets.scannet import ScanNet_Multi  # noqa: F401

from streamvggt.datasets.config import (  # noqa: F401
    DATASET_REGISTRY,
    DatasetConfig,
    DatasetName,
    Split,
    build_dataset,
)


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length,
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    return data_loader
