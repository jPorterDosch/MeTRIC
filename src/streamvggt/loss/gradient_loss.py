import torch

from .trimmed_loss import TrimmedMAELoss
from .utils import gradient_loss, reduction_batch_based, reduction_image_based


class GradientLoss(torch.nn.Module):
    def __init__(self, scales: int = 4, reduction: str = "batch-based") -> None:
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        num_frame_h: int = 1,
    ) -> torch.Tensor:
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i * frame_h : (i + 1) * frame_h, :] = i + 1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step]
                if num_frame_h > 1
                else None,
            )

        return total


class TemporalGradientMatchingLoss(torch.nn.Module):
    def __init__(
        self,
        trim: float = 0.2,
        temp_grad_scales: int = 4,
        temp_grad_decay: float = 0.5,
        reduction: str = "batch-based",
        diff_depth_th: float = 0.05,
    ) -> None:
        super().__init__()

        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        prediction: Shape(B, T, H, W)
        target: Shape(B, T, H, W)
        mask: Shape(B, T, H, W)
        """
        total = 0
        cnt = 0

        min_target = (
            torch.where(mask.bool(), target, torch.inf).min(-1).values.min(-1).values
        )
        max_target = (
            torch.where(mask.bool(), target, -torch.inf).max(-1).values.max(-1).values
        )
        target_th = (max_target - min_target) * self.diff_depth_th

        for scale in range(self.temp_grad_scales):
            temp_stride = pow(2, scale)
            if temp_stride < prediction.shape[1]:
                pred_temp_grad = torch.diff(prediction[:, ::temp_stride, ...], dim=1)
                target_temp_grad = torch.diff(target[:, ::temp_stride, ...], dim=1)
                temp_mask = (
                    mask[:, ::temp_stride, ...][:, 1:, ...]
                    & mask[:, ::temp_stride, ...][:, :-1, ...]
                )

                valid_mask_from_target_th = (
                    target_temp_grad.abs()
                    < target_th.unsqueeze(-1).unsqueeze(-1)[:, ::temp_stride, ...][
                        :, 1:, ...
                    ]
                )
                temp_mask = temp_mask & valid_mask_from_target_th

                total += self.data_loss(
                    prediction=pred_temp_grad.flatten(0, 1),
                    target=target_temp_grad.flatten(0, 1),
                    mask=temp_mask.flatten(0, 1),
                ) * pow(self.temp_grad_decay, scale)
                cnt += 1

        return total / cnt
