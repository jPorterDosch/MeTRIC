"""Enums for the streamvggt dataset pipeline.

Kept in a dependency-free leaf module so both the dataset loaders and the
config can import them without a cycle. Members subclass ``(str, enum.Enum)``
so a plain string from a CLI / YAML / test coerces to a member via
``Split(value)``, matching the house style in
``streamvggt.depth_cond.config``.
"""

import enum


class Split(str, enum.Enum):
    TRAIN = "train"
    TEST = "test"


class DatasetName(str, enum.Enum):
    HAMMER = "hammer"
    # ARKitScenes ships as two disjoint variants that partition the scenes:
    # LOWRES = LiDAR depth + pairs-based sampling (new_scene_metadata.npz),
    # HIGHRES = laser-scanner GT depth + timestamp sampling (scene_metadata.npz).
    # There is deliberately no bare "arkitscenes" member: selecting a variant
    # is part of the experiment identity, so it must be explicit.
    ARKITSCENES_LOWRES = "arkitscenes_lowres"
    ARKITSCENES_HIGHRES = "arkitscenes_highres"
    SCANNET = "scannet"


class TransformName(str, enum.Enum):
    IMGNORM = "imgnorm"
    SEQ_COLOR_JITTER = "seq_color_jitter"
    COLOR_JITTER = "color_jitter"
