"""Leaf utilities for the streamvggt dataset pipeline (image IO, geometry,
cropping, correspondences, transforms). Self-contained: no dependency on the
dust3r package tree.

Re-exports the public helpers so callers can import from the package directly,
e.g. ``from streamvggt.datasets.utils import imread_cv2, ImgNorm, geotrf``.
The submodules (``cropping``, ``corr``) remain importable as
``streamvggt.datasets.utils.cropping`` for their lower-level entry points.
"""

from .device import to_numpy, todevice
from .geometry import (
    colmap_to_opencv_intrinsics,
    depthmap_to_absolute_camera_coordinates,
    depthmap_to_camera_coordinates,
    geotrf,
    inv,
    opencv_to_colmap_intrinsics,
)
from .image import ImgNorm, imread_cv2, rgb
from .transforms import ColorJitter, SeqColorJitter

__all__ = [
    # device
    "to_numpy",
    "todevice",
    # geometry
    "geotrf",
    "inv",
    "depthmap_to_camera_coordinates",
    "depthmap_to_absolute_camera_coordinates",
    "colmap_to_opencv_intrinsics",
    "opencv_to_colmap_intrinsics",
    # image
    "ImgNorm",
    "imread_cv2",
    "rgb",
    # transforms
    "ColorJitter",
    "SeqColorJitter",
]
