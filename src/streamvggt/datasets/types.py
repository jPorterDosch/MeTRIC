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
    ARKITSCENES = "arkitscenes"
    SCANNET = "scannet"


class TransformName(str, enum.Enum):
    IMGNORM = "imgnorm"
    SEQ_COLOR_JITTER = "seq_color_jitter"
    COLOR_JITTER = "color_jitter"
