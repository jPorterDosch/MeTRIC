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
        # GT dense depth lives in "depthmap" [B,H,W]; the model's prediction is
        # "depth" [B,H,W,1] ("depth" on a GT view would be a teacher output, not
        # ground truth -- cf. FinetuneLoss, which also reads g["depthmap"]).
        pred_depth = torch.stack([p["depth"] for p in preds], dim=1).squeeze(-1)
        gt_depth = torch.stack([g["depthmap"] for g in gts], dim=1)
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
        # per-frame breakdown of Ldepth for logging: main (confidence-weighted
        # L1), grad (spatial-gradient/edge), reg (-alpha*log(sigma) confidence
        # regularizer). They sum to Ldepth; logged separately so the metric-mode
        # balance between the accuracy term and the regularizer is visible.
        comp_terms: dict[str, list[torch.Tensor]] = {"main": [], "grad": [], "reg": []}
        for g, p in zip(gts, preds, strict=True):
            if "depth" in p:
                sigma_p = p["depth_conf"]
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    valid_mask = torch.ones_like(g["valid_mask"])
                term, comps = self.depth_loss(
                    p["depth"],
                    g["depthmap"].unsqueeze(-1),
                    sigma_p=sigma_p,
                    valid_mask=valid_mask,
                    return_components=True,
                )
                depth_terms.append(term)
                for k, v in comps.items():
                    comp_terms[k].append(v)
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

        # logging-only breakdown of Ldepth (NOT added to total; detached in
        # DepthOrPmapLoss). Zeros when no frame carried a depth prediction,
        # mirroring Ldepth's own fallback.
        for name, terms in comp_terms.items():
            details[f"Ldepth_{name}"] = (
                torch.stack(terms).mean() if terms else torch.zeros_like(Ltemporal)
            )

        return total, details
