import torch
from .gradient_loss import TemporalGradientMatchingLoss
from .trimmed_loss import TrimmedProcrustesLoss
from .utils import compute_scale_and_shift


class VideoDepthLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        scales: int = 4,
        trim: float = 0.0,
        stable_scale: float = 10,
        reduction: str = "batch-based",
    ) -> None:
        super().__init__()
        self.spatial_loss = TrimmedProcrustesLoss(
            alpha=alpha, scales=scales, trim=trim, reduction=reduction
        )
        self.stable_loss = TemporalGradientMatchingLoss(
            trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1
        )
        self.stable_scale = stable_scale

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        prediction: Shape(B, T, H, W)
        target: Shape(B, T, H, W)
        mask: Shape(B, T, H, W)
        """
        loss_dict = {}
        total = 0
        loss_dict["spatial_loss"] = self.spatial_loss(
            prediction=prediction.flatten(0, 1),
            target=target.flatten(0, 1),
            mask=mask.flatten(0, 1).float(),
        )
        total += loss_dict["spatial_loss"]
        scale, shift = compute_scale_and_shift(
            prediction.flatten(1, 2), target.flatten(1, 2), mask.flatten(1, 2)
        )
        prediction = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)
        loss_dict["stable_loss"] = (
            self.stable_loss(prediction=prediction, target=target, mask=mask)
            * self.stable_scale
        )
        total += loss_dict["stable_loss"]

        loss_dict["total_loss"] = total
        return loss_dict
