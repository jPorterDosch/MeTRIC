import torch
from typing import Any

from .utils import (
    normalize_prediction_robust,
    reduction_batch_based,
    reduction_image_based,
)


class TrimmedProcrustesLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        scales: int = 4,
        trim: float = 0.2,
        reduction: str = "batch-based",
    ) -> None:
        super().__init__()

        # Local import: gradient_loss imports TrimmedMAELoss from this module, so
        # a top-level import here would form a cycle. Only TrimmedProcrustesLoss
        # needs GradientLoss, and only at construction, so defer it to here.
        from .gradient_loss import GradientLoss

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi: torch.Tensor | None = None
        self.__prediction_median_scale: tuple[Any, Any] | None = None
        self.__target_median_scale: tuple[Any, Any] | None = None

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pred_ms: tuple[torch.Tensor, torch.Tensor] | None = None,
        tar_ms: tuple[torch.Tensor, torch.Tensor] | None = None,
        num_frame_h: int = 1,
        no_norm: bool = False,
    ) -> torch.Tensor:
        if no_norm:
            self.__prediction_ssi, self.__prediction_median_scale = prediction, (0, 1)
            target_, self.__target_median_scale = target, (0, 1)
        else:
            self.__prediction_ssi, self.__prediction_median_scale = (
                normalize_prediction_robust(prediction, mask, ms=pred_ms)
            )
            target_, self.__target_median_scale = normalize_prediction_robust(
                target, mask, ms=tar_ms
            )

        total = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask, num_frame_h=num_frame_h
            )

        return total

    def get_median_scale(self) -> tuple[Any, Any]:
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self) -> torch.Tensor | None:
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(torch.nn.Module):
    def __init__(self, trim: float = 0.2, reduction: str = "batch-based") -> None:
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weight_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[:keep_num]

        return self.__reduction(trimmed, M)
