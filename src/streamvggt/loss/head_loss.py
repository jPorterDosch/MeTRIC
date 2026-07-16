import torch
import torch.nn.functional as F

from .utils import (
    check_and_fix_inf_nan,
    closed_form_scale_and_shift,
    normalize_pointcloud,
    point_map_to_normal,
)


class CameraLoss(torch.nn.Module):
    def __init__(
        self, delta: float = 1e-1, weights: tuple[float, float, float] = (1.0, 1.0, 0.5)
    ) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        loss_T = (pred_pose[..., :3] - gt_pose[..., :3]).abs()
        loss_R = (pred_pose[..., 3:7] - gt_pose[..., 3:7]).abs()
        loss_FL = (pred_pose[..., 7:] - gt_pose[..., 7:]).abs()

        loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
        loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
        loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

        # Clamp outlier translation loss to prevent instability, then average
        loss_T = loss_T.clamp(max=100).mean()
        loss_R = loss_R.mean()
        loss_FL = loss_FL.mean()
        return (
            self.weights[0] * loss_T
            + self.weights[1] * loss_R
            + self.weights[2] * loss_FL
        )


class DepthOrPmapLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.01,
        metric: bool = False,
        log_space: bool = False,
        log_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.grad_scales = 3
        self.gamma = 1.0
        # metric=True supervises absolute depth: skip the per-sample
        # normalize_pointcloud + closed-form scale/shift alignment so the loss
        # penalizes getting the metric scale/offset wrong (used when the model
        # is conditioned on real metric depth). metric=False keeps the default
        # scale-and-shift-invariant behavior.
        self.metric = metric
        # log_space=True computes the accuracy AND gradient terms on log-depth,
        # so |log pred - log gt| ~ |pred-gt|/gt (relative error). This stops the
        # far background from dominating the metric L1 while KEEPING the scale
        # penalty (a wrong global scale is a constant log offset -- unlike
        # scale-invariant SILog, which subtracts the mean and would discard the
        # metric signal this model is conditioned for). Depth-head only (the
        # log is on the [B,H,W,1] depth, not the pointmap branch).
        self.log_space = log_space
        self.log_eps = log_eps

    def gradient_loss_multi_scale(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        total = 0
        for s in range(self.grad_scales):
            step = 2**s
            pred_s = pred[:, ::step, ::step]
            gt_s = gt[:, ::step, ::step]
            mask_s = mask[:, ::step, ::step]
            total += self.normal_loss(pred_s, gt_s, mask_s)
        return total / self.grad_scales

    def normal_loss(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred_norm, _ = point_map_to_normal(pred, mask)
        gt_norm, _ = point_map_to_normal(gt, mask)
        cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=-1)
        return 1 - cos_sim.mean()

    def image_gradient_loss(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert pred.dim() == 4 and pred.shape[-1] == 1
        assert gt.shape == pred.shape

        B, H, W, _ = pred.shape
        _ = pred.device

        dx_pred = pred[:, :, 1:] - pred[:, :, :-1]  # [B,H,W-1,1]
        dx_gt = gt[:, :, 1:] - gt[:, :, :-1]
        dx_mask = mask[:, :, 1:] & mask[:, :, :-1]  # [B,H,W-1]

        dy_pred = pred[:, 1:, :] - pred[:, :-1, :]  # [B,H-1,W,1]
        dy_gt = gt[:, 1:, :] - gt[:, :-1, :]
        dy_mask = mask[:, 1:, :] & mask[:, :-1, :]  # [B,H-1,W]

        min_h = min(dy_pred.shape[1], dx_pred.shape[1])
        min_w = min(dx_pred.shape[2], dy_pred.shape[2])

        dx_pred = dx_pred[:, :min_h, :min_w, :]  # [B,H-1,W-1,1]
        dx_gt = dx_gt[:, :min_h, :min_w, :]
        dx_mask = dx_mask[:, :min_h, :min_w]  # [B,H-1,W-1]

        dy_pred = dy_pred[:, :min_h, :min_w, :]  # [B,H-1,W-1,1]
        dy_gt = dy_gt[:, :min_h, :min_w, :]
        dy_mask = dy_mask[:, :min_h, :min_w]  # [B,H-1,W-1]

        loss_dx = F.l1_loss(
            dx_pred * dx_mask.unsqueeze(-1), dx_gt * dx_mask.unsqueeze(-1)
        )
        loss_dy = F.l1_loss(
            dy_pred * dy_mask.unsqueeze(-1), dy_gt * dy_mask.unsqueeze(-1)
        )

        return (loss_dx + loss_dy) / 2

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        sigma_p: torch.Tensor | None = None,
        sigma_g: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.metric:
            # Metric supervision: compare raw predicted depth against raw GT,
            # with no normalization or scale/shift fit, so absolute scale is
            # penalized.
            pred_aligned, gt_normalized = pred, gt
        else:
            if self.training:
                pred_normalized, _ = normalize_pointcloud(pred, valid_mask)
                gt_normalized, _ = normalize_pointcloud(gt, valid_mask)
            else:
                pred_normalized, gt_normalized = pred, gt
            scale, shift = closed_form_scale_and_shift(pred_normalized, gt_normalized)
            pred_aligned = pred_normalized * scale + shift
        sigma_p = sigma_p.clamp(min=1e-6)
        if sigma_g is not None:
            sigma_g = sigma_g.clamp(min=1e-6)
        # sigma = 0.5 * (sigma_p + sigma_g)
        sigma = sigma_p

        # compare in log-depth (relative, scale-aware) when requested. clamp
        # keeps the log finite where pred is <=0 (the inv_log head can emit
        # small negatives) or gt is 0 (invalid); those pixels are dropped by
        # valid_mask before the mean, and the clamp only stops NaN/inf from
        # leaking through the masked gradient product.
        if self.log_space and pred.shape[-1] == 1:
            pred_cmp = torch.log(pred_aligned.clamp(min=self.log_eps))
            gt_cmp = torch.log(gt_normalized.clamp(min=self.log_eps))
        else:
            pred_cmp, gt_cmp = pred_aligned, gt_normalized

        diff = (pred_cmp - gt_cmp).abs()

        C = diff.shape[-1]

        main_loss = (sigma[..., None].expand(-1, -1, -1, C) * diff)[
            valid_mask[..., None].expand(-1, -1, -1, C)
        ].mean()

        if pred.shape[-1] == 1:
            grad_loss = self.image_gradient_loss(pred_cmp, gt_cmp, valid_mask)
        else:
            grad_loss = self.gradient_loss_multi_scale(pred_cmp, gt_cmp, valid_mask)
        reg_loss = -self.alpha * torch.log(sigma.clamp(min=1e-6))[valid_mask].mean()
        main_term = self.gamma * main_loss
        total = main_term + grad_loss + reg_loss
        if return_components:
            # detached copies for logging only; `total` keeps its graph so the
            # caller still backprops through the full summed loss. main =
            # confidence-weighted L1, grad = spatial-gradient/edge, reg =
            # -alpha*log(sigma) confidence regularizer. They sum to `total`.
            #
            # main_raw and conf DECOMPOSE main = conf-weight x error: main_raw is
            # the sigma-FREE masked error (tracks true accuracy, should mirror
            # AbsRel), conf is mean sigma (the weight). If main rises on val while
            # main_raw stays flat, the overfit is confidence, not depth accuracy.
            # Neither is in `total` (both detached, logging only).
            vm = valid_mask[..., None].expand(-1, -1, -1, C)
            main_raw = diff[vm].mean()
            conf = sigma[valid_mask].mean()
            return total, {
                "main": main_term.detach(),
                "grad": grad_loss.detach(),
                "reg": reg_loss.detach(),
                "main_raw": main_raw.detach(),
                "conf": conf.detach(),
            }
        return total


class TrackLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = 0.2
        self.gamma = 1.0

    def forward(
        self,
        y_pr: torch.Tensor,
        y_gt: torch.Tensor,
        vis_pr: torch.Tensor,
        vis_gt: torch.Tensor,
        w_p: torch.Tensor,
        w_g: torch.Tensor,
    ) -> torch.Tensor:
        # w = 0.5 * (w_p + w_g)
        w = w_p
        l_pos = (y_pr - y_gt).norm(dim=-1)
        l_pos = (w * l_pos).mean()

        l_vis = self.bce(vis_pr, vis_gt.float())
        l_vis = (w * l_vis).mean()
        return l_pos + l_vis
