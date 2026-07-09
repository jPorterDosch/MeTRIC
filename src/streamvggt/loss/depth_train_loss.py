import torch

from .base import Details, MultiLoss, View
from .gradient_loss import TemporalGradientMatchingLoss
from .head_loss import DepthOrPmapLoss
from .utils import compute_scale_and_shift


class DepthTrainLoss(MultiLoss):
    def __init__(
        self,
        weights: tuple[float, ...] | None = None,
        alpha: float = 0.1,
        trim: float = 0.2,
        temp_grad_scales: int = 4,
        temp_grad_decay: float = 0.5,
        reduction: str = "batch-based",
        diff_depth_th: float = 0.05,
        metric: bool = False,
    ) -> None:
        super().__init__()
        # metric=True supervises absolute depth end-to-end: the depth term skips
        # its scale/shift alignment and the temporal term skips its scale fit, so
        # the metric scale the model is conditioned on is actually penalized.
        self.metric = metric
        self.temporal_loss = TemporalGradientMatchingLoss(
            trim=trim,
            temp_grad_scales=temp_grad_scales,
            temp_grad_decay=temp_grad_decay,
            reduction=reduction,
            diff_depth_th=diff_depth_th,
        )
        self.depth_loss = DepthOrPmapLoss(alpha=alpha, metric=metric)

        if weights is None:
            # equal weight per objective (temporal, depth) when none supplied
            weights = (1.0,) * 2

        self.weights = weights

    def get_name(self) -> str:
        return "DepthTrainLoss"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        losses = []

        # ---------- Ltemporal (temporal depth-gradient consistency) ----------
        pred_depth = torch.stack([p["depth"] for p in preds], dim=1).squeeze(-1)
        gt_depth = torch.stack([g["depth"] for g in gts], dim=1).squeeze(-1)
        temp_mask = torch.stack([g["valid_mask"] for g in gts], dim=1)

        if not self.metric:
            # Align prediction to GT scale/shift over the clip before matching
            # temporal gradients (cf. VideoDepthLoss); skipped in metric mode so
            # the absolute temporal rate-of-change is supervised. (The +shift is
            # a no-op for the temporal diff, but the *scale removal is what we
            # drop for metric.)
            scale, shift = compute_scale_and_shift(
                pred_depth.flatten(1, 2),
                gt_depth.flatten(1, 2),
                temp_mask.flatten(1, 2),
            )
            pred_depth = scale.view(-1, 1, 1, 1) * pred_depth + shift.view(-1, 1, 1, 1)
        Ltemporal = self.temporal_loss(
            prediction=pred_depth, target=gt_depth, mask=temp_mask
        )

        losses.append((Ltemporal, "Ltemporal"))

        # ---------- Ldepth ----------
        depth_terms = []
        for g, p in zip(gts, preds):
            if ("depth" in g) and ("depth" in p):
                sigma_p = p["depth_conf"]
                sigma_g = g["depth_conf"]
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    valid_mask = torch.ones_like(g["valid_mask"])
                depth_terms.append(
                    self.depth_loss(
                        p["depth"], g["depth"], sigma_p, sigma_g, valid_mask
                    )
                )
        Ldepth = (
            torch.stack(depth_terms).mean()
            if depth_terms
            else torch.zeros_like(Ltemporal)
        )

        losses.append((Ldepth, "Ldepth"))

        total = 0.0
        details = {}

        for loss, weight in zip(losses, self.weights, strict=True):
            total += weight * loss[0]
            details[loss[1]] = loss[0]

        return total, details
