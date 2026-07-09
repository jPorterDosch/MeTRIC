from __future__ import annotations
from copy import copy, deepcopy
import torch

from typing import Any

# a per-view sample: image/depth/pose/mask tensors keyed by field name
View = dict[str, torch.Tensor]
# heterogeneous per-loss logging payload (tensors, floats, lists, ...)
Details = dict[str, Any]
# (translation, quaternion) pair, each shaped (..., 3) / (..., 4)
Pose = tuple[torch.Tensor, torch.Tensor]


class BaseCriterion(torch.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


class Criterion(torch.nn.Module):
    def __init__(self, criterion: BaseCriterion | None = None) -> None:
        super().__init__()
        if not isinstance(criterion, BaseCriterion):
            raise ValueError(f"{criterion} is not a proper criterion!")
        self.criterion = copy(criterion)

    def get_name(self) -> str:
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode: str = "none") -> Criterion:
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(torch.nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha = 1
        self._loss2: MultiLoss | None = None

    def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()

    def __mul__(self, alpha: int | float) -> MultiLoss:
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2: MultiLoss) -> MultiLoss:
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self) -> str:
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Details]:
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details
