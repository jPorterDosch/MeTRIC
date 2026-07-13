from .base import BaseCriterion
import torch


class LLoss(BaseCriterion):
    """L-norm loss"""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, (
            f"Bad shape = {a.shape}"
        )
        dist = self.distance(a, b)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


class MSELoss(LLoss):
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) ** 2


L21 = L21Loss()
MSE = MSELoss()
