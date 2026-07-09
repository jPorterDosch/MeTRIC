import torch
from typing import Any

from .base import BaseCriterion, Criterion, Details, MultiLoss, View
from .ssim import SSIM


class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion: BaseCriterion) -> None:
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.criterion(a, b)

    def compute_loss(
        self, gts: list[View], preds: list[View], **kw: Any
    ) -> tuple[torch.Tensor, Details]:
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, loss in enumerate(ls):
            details[self_name + f"_rgb/{i + 1}"] = float(loss)
            details[f"pred_rgb_{i + 1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details
