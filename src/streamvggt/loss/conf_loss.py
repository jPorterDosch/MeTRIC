import torch
from typing import Any

from .base import Details, MultiLoss, View


class ConfLoss(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss: MultiLoss, alpha: float = 1) -> None:
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self) -> str:
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.log(x)

    def compute_loss(
        self, gts: list[View], preds: list[View], **kw: Any
    ) -> tuple[torch.Tensor, Details]:
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        if "is_self" in details and "img_ids" in details:
            img_ids = details["img_ids"]
        else:
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf"

            camera_only = gts[0]["camera_only"]
            conf, log_conf = self.get_conf_log(
                pred[conf_key][~camera_only][losses_and_masks[i][1]]
            )

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            self_name = type(self).__name__
            details[self_name + f"_conf_loss/{img_ids[i] + 1}"] = float(conf_loss)

        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            final_loss = (
                final_loss + details["pose_loss"].clip(max=0.3) * 5.0
            )  # , details
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details
