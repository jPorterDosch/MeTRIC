"""Small training utilities for the depth-conditioning entrypoint."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch (CPU + CUDA), numpy, and the stdlib RNG."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
