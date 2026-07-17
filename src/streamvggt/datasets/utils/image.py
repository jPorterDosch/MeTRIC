# Image IO / normalization helpers for the streamvggt dataset pipeline.
#
# Self-contained subset copied from DUSt3R (dust3r.utils.image): the normalized
# tensor transform and the opencv reader used by every dataset loader.
import os

import numpy as np
import torch
import torchvision.transforms as tvf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402

from .zipio import read_bytes  # noqa: E402

# maps a PIL image (or HxWx3 uint8 array) into a [-1, 1] float tensor
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python, from disk or from
    inside a stored scene zip (see zipio.read_bytes for the '<scene>.zip/'
    virtual-path convention)."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    if ".zip/" in path:
        img = cv2.imdecode(np.frombuffer(read_bytes(path), np.uint8), options)
    else:
        img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    """Undo ImgNorm: turn a normalized tensor back into a [0, 1] HxWx3 array."""
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)
