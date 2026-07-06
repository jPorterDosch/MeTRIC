from __future__ import annotations

from copy import copy, deepcopy
from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dust3r.utils.geometry import (
    inv,
    geotrf,
    normalize_pointcloud_group,
    get_group_pointcloud_center_scale,
)
from dust3r.utils.camera import (
    camera_to_pose_encoding,
    relative_pose_absT_quatR,
)
from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding

# a per-view sample: image/depth/pose/mask tensors keyed by field name
View = dict[str, torch.Tensor]
# heterogeneous per-loss logging payload (tensors, floats, lists, ...)
Details = dict[str, Any]
# (translation, quaternion) pair, each shaped (..., 3) / (..., 4)
Pose = tuple[torch.Tensor, torch.Tensor]


def Sum(
    *losses_and_masks: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor | tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


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


L21 = L21Loss()


class MSELoss(LLoss):
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) ** 2


MSE = MSELoss()


class Criterion(nn.Module):
    def __init__(self, criterion: BaseCriterion | None = None) -> None:
        super().__init__()
        assert isinstance(criterion, BaseCriterion), (
            f"{criterion} is not a proper criterion!"
        )
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


class MultiLoss(nn.Module):
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


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self) -> None:
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


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
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i + 1}"] = float(l)
            details[f"pred_rgb_{i + 1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction: str = "none") -> None:
        super().__init__(reduction)

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction: str = "none") -> None:
        super().__init__(reduction)

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion: BaseCriterion,
        norm_mode: str = "?avg_dis",
        gt_scale: bool = False,
        sky_loss_value: float = 2,
        max_metric_scale: bool | float = False,
    ) -> None:
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self,
        pts_cross: list[torch.Tensor],
        valids: list[torch.Tensor],
        conf_cross: list[torch.Tensor],
        norm_self_only: bool = False,
    ) -> torch.Tensor:
        pts = [x for x in pts_cross]
        valids = [x for x in valids]
        confs = [x for x in conf_cross]
        norm_factor = normalize_pointcloud_group(
            pts, self.norm_mode, valids, confs, ret_factor_only=True
        )
        return norm_factor

    def get_norm_factor_poses(
        self,
        gt_trans: list[torch.Tensor],
        pr_trans: list[torch.Tensor],
        not_metric_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.norm_mode and not self.gt_scale:
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
        else:
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric
        return norm_factor_gt, norm_factor_pr

    def get_all_pts3d(
        self,
        gts: list[View],
        preds: list[View],
        dist_clip: float | None = None,
        norm_self_only: bool = False,
        norm_pose_separately: bool = False,
        eps: float = 1e-3,
        camera1: torch.Tensor | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[Pose],
        list[Pose],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        Details,
    ]:
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        # valids = torch.stack(valids, dim=0)  # S B H W
        # valids = valids.permute(1, 0, 2, 3)  # B S H W
        # valids_masks = preprocess_mask(valids, mode="pad") # (B, S, H, W)
        #
        # valids = torch.unbind(valids_masks, dim=1) # [S] (B, H, W)

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_cross,
                valids,
                conf_cross,
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )

        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:7]) for pr in pr_poses
        ]
        pose_masks = (pose_norm_factor_gt.squeeze(-1) > eps) & (
            pose_norm_factor_pr.squeeze(-1) > eps
        )

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {},
        )

    def get_all_pts3d_with_scale_loss(
        self,
        gts: list[View],
        preds: list[View],
        dist_clip: float | None = None,
        norm_self_only: bool = False,
        norm_pose_separately: bool = False,
        eps: float = 1e-3,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[Pose],
        list[Pose],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        Details,
    ]:
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self[:1],
                gt_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        if self.norm_mode:
            norm_factor_pr = self.get_norm_factor_point_cloud(
                pr_pts_self[:1],
                pr_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            raise NotImplementedError
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pts_scale_loss = torch.abs(
                norm_factor_pr[~not_metric_mask] - norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pts_scale_loss = 0.0

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, torch.ones_like(not_metric_mask)
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(
                    gt_trans, pr_trans, torch.ones_like(not_metric_mask)
                )
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pose_scale_loss = torch.abs(
                pose_norm_factor_pr[~not_metric_mask]
                - pose_norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pose_scale_loss = 0.0
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:7]) for pr in pr_poses
        ]

        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {"scale_loss": pose_scale_loss + pts_scale_loss},
        )

    def compute_relative_pose_loss(
        self,
        gt_trans: torch.Tensor,
        gt_quats: torch.Tensor,
        pr_trans: torch.Tensor,
        pr_quats: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if masks is None:
            masks = torch.ones(len(gt_trans), dtype=torch.bool, device=gt_trans.device)
        gt_trans_matrix1 = gt_trans[:, :, None, :].repeat(1, 1, gt_trans.shape[1], 1)[
            masks
        ]
        gt_trans_matrix2 = gt_trans[:, None, :, :].repeat(1, gt_trans.shape[1], 1, 1)[
            masks
        ]
        gt_quats_matrix1 = gt_quats[:, :, None, :].repeat(1, 1, gt_quats.shape[1], 1)[
            masks
        ]
        gt_quats_matrix2 = gt_quats[:, None, :, :].repeat(1, gt_quats.shape[1], 1, 1)[
            masks
        ]
        pr_trans_matrix1 = pr_trans[:, :, None, :].repeat(1, 1, pr_trans.shape[1], 1)[
            masks
        ]
        pr_trans_matrix2 = pr_trans[:, None, :, :].repeat(1, pr_trans.shape[1], 1, 1)[
            masks
        ]
        pr_quats_matrix1 = pr_quats[:, :, None, :].repeat(1, 1, pr_quats.shape[1], 1)[
            masks
        ]
        pr_quats_matrix2 = pr_quats[:, None, :, :].repeat(1, pr_quats.shape[1], 1, 1)[
            masks
        ]

        gt_rel_trans, gt_rel_quats = relative_pose_absT_quatR(
            gt_trans_matrix1, gt_quats_matrix1, gt_trans_matrix2, gt_quats_matrix2
        )
        pr_rel_trans, pr_rel_quats = relative_pose_absT_quatR(
            pr_trans_matrix1, pr_quats_matrix1, pr_trans_matrix2, pr_quats_matrix2
        )
        rel_trans_err = torch.norm(gt_rel_trans - pr_rel_trans, dim=-1)
        rel_quats_err = torch.norm(gt_rel_quats - pr_rel_quats, dim=-1)
        return rel_trans_err.mean() + rel_quats_err.mean()

    def compute_pose_loss(
        self,
        gt_poses: list[Pose],
        pred_poses: list[Pose],
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B
        """
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BXNX3
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx4
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4
        if masks == None:
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
        else:
            if not any(masks):
                return torch.tensor(0.0)
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )

        return pose_loss

    def compute_loss(
        self, gts: list[View], preds: list[View], **kw: Any
    ) -> tuple[torch.Tensor, Details]:
        (
            gt_pts_cross,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "sky_loss_value should be 0 if no conf loss"
            )
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        # if self.sky_loss_value > 0:
        #     assert (
        #         self.criterion.reduction == "none"
        #     ), "sky_loss_value should be 0 if no conf loss"
        #     for i, l in enumerate(ls_self):
        #         ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}

        # cross view loss and details
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]

        for i in range(len(ls_cross)):
            details[f"gt_img{i + 1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i + 1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i + 1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i + 1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i + 1}"] = preds[i]["desc"].detach()

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "sky_loss_value should be 0 if no conf loss"
            )
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i + 1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i + 1}"] = preds[i]["conf"].detach()

        ls = ls_cross
        masks = masks_cross
        details["img_ids"] = np.arange(len(ls_cross)).tolist()
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPoseBatchList(Regr3DPose):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion: BaseCriterion,
        norm_mode: str = "?avg_dis",
        gt_scale: bool = False,
        sky_loss_value: float = 2,
        max_metric_scale: bool | float = False,
    ) -> None:
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )
        self.depth_only_criterion = DepthScaleShiftInvLoss()
        self.single_view_criterion = ScaleInvLoss()

    def reorg(
        self, ls_b: list[torch.Tensor], masks_b: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]
        ls = [[] for _ in range(len(masks_b[0]))]
        for i in range(len(ls_b)):
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(
        self, gts: list[View], preds: list[View], **kw: Any
    ) -> tuple[torch.Tensor, Details]:
        (
            gt_pts_cross,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "sky_loss_value should be 0 if no conf loss"
            )
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        camera_only = gts[0]["camera_only"]
        depth_only = gts[0]["depth_only"]
        single_view = gts[0]["single_view"]
        is_metric = gts[0]["is_metric"]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError
        else:
            # list [(B, h, w, 3)] x num_views -> list [num_views, h, w, 3] x B
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)

        self_name = type(self).__name__

        gt_pts_cross_b = torch.unbind(
            torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
        )
        pred_pts_cross_b = torch.unbind(
            torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
        )
        masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)
        ls_cross_b = []
        for i in range(len(gt_pts_cross_b)):
            if depth_only[~camera_only][i]:
                ls_cross_b.append(
                    self.depth_only_criterion(
                        pred_pts_cross_b[i][..., -1],
                        gt_pts_cross_b[i][..., -1],
                        masks_cross_b[i],
                    )
                )
            elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                ls_cross_b.append(
                    self.single_view_criterion(
                        pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                    )
                )
            else:
                ls_cross_b.append(
                    self.criterion(
                        pred_pts_cross_b[i][masks_cross_b[i]],
                        gt_pts_cross_b[i][masks_cross_b[i]],
                    )
                )
        ls_cross = self.reorg(ls_cross_b, masks_cross_b)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == "none", (
                "sky_loss_value should be 0 if no conf loss"
            )
            masks_cross = [mask[~camera_only] for mask in masks]
            skys_cross = [sky[~camera_only] for sky in skys]
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        details = {}
        for i in range(len(ls_cross)):
            details[f"gt_img{i + 1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i + 1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i + 1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i + 1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i + 1}"] = preds[i]["desc"].detach()

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i + 1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i + 1}"] = preds[i]["conf"].detach()

        ls = ls_cross
        masks = masks_cross
        details["img_ids"] = np.arange(len(ls_cross)).tolist()
        pose_masks = pose_masks * gts[i]["img_mask"]
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


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

            details[self.get_name() + f"_conf_loss/{img_ids[i] + 1}"] = float(conf_loss)

        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            final_loss = (
                final_loss + details["pose_loss"].clip(max=0.3) * 5.0
            )  # , details
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details


class Regr3DPose_ScaleInv(Regr3DPose):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(
        self, gts: list[View], preds: list[View]
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[Pose],
        list[Pose],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        Details,
    ]:
        # compute depth-normalized points
        (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = super().get_all_pts3d(gts, preds)

        # measure scene scale

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        return (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        )


def reduction_batch_based(image_loss: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = reduction_batch_based,
    frame_id_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # mask for distinguish different frames
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = (
            (frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0
        ).to(mask.dtype)
        valid_id_mask_y = (
            (frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0
        ).to(mask.dtype)

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def normalize_prediction_robust(
    target: torch.Tensor,
    mask: torch.Tensor,
    ms: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)

        m[valid] = torch.median(
            (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
        ).values
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())


def compute_scale_and_shift(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (
        det[valid] + 1e-6
    )
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (
        det[valid] + 1e-6
    )

    return x_0, x_1


class TrimmedProcrustesLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        scales: int = 4,
        trim: float = 0.2,
        reduction: str = "batch-based",
    ) -> None:
        super().__init__()

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


class TrimmedMAELoss(nn.Module):
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


class GradientLoss(nn.Module):
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


class TemporalGradientMatchingLoss(nn.Module):
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


class VideoDepthLoss(nn.Module):
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


def closed_form_scale_and_shift(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        pred:   (B, H, W, C)
        gt:     (B, H, W, C)
        valid_mask: (B, H, W)
    Returns:
        scale:  (B,)
        shift:  (B,)
    """
    assert pred.dim() == 4 and gt.dim() == 4, "Inputs must be 4D tensors"
    B, H, W, C = pred.shape
    device = pred.device

    pred_flat = pred.view(-1, C)  # (N, C)
    gt_flat = gt.view(-1, C)  # (N, C)

    if C == 1:
        pred_mean = pred_flat.mean(dim=0)
        gt_mean = gt_flat.mean(dim=0)

        numerator = ((pred_flat - pred_mean) * (gt_flat - gt_mean)).sum(dim=0)
        denominator = ((pred_flat - pred_mean) ** 2).sum(dim=0).clamp(min=1e-6)
        scale = numerator / denominator

        shift = gt_mean - scale * pred_mean
        return scale, shift

    elif C == 3:
        pred_mean = pred_flat.mean(0)
        gt_mean = gt_flat.mean(0)
        pred_centered = pred_flat - pred_mean
        gt_centered = gt_flat - gt_mean

        scale = (pred_centered * gt_centered).sum() / (pred_centered**2).sum().clamp(
            min=1e-6
        )
        shift = gt_mean - scale * pred_mean
        return scale, shift

    else:
        raise ValueError(
            f"Unsupported channel dimension C={C}. Only 1 or 3 channels are supported."
        )


def normalize_pointcloud(
    pts3d: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pts3d: B, H, W, 3
    valid_mask: B, H, W
    """
    dist = pts3d.norm(dim=-1)
    dist_sum = (dist * valid_mask).sum(dim=[1, 2])
    valid_count = valid_mask.sum(dim=[1, 2])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1)
    return pts3d, avg_scale


def point_map_to_normal(
    point_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        pts = F.pad(
            point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0
        ).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

        v1 = (
            padded_mask[:, :-2, 1:-1]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 1:-1, :-2]
        )
        v2 = (
            padded_mask[:, 1:-1, :-2]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 2:, 1:-1]
        )
        v3 = (
            padded_mask[:, 2:, 1:-1]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 1:-1, 2:]
        )
        v4 = (
            padded_mask[:, 1:-1, 2:]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, :-2, 1:-1]
        )

        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def check_and_fix_inf_nan(
    tensor: torch.Tensor, name: str, hard_max: float = 100
) -> torch.Tensor:
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    if invalid_mask.any():
        print(
            f"[warning] {name} contains {invalid_mask.sum().item()} inf/nan values, replacing with 0"
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=hard_max, neginf=-hard_max)
    return tensor


class CameraLoss(nn.Module):
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


class DepthOrPmapLoss(nn.Module):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.grad_scales = 3
        self.gamma = 1.0

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
        device = pred.device

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
    ) -> torch.Tensor:
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
        diff = (pred_aligned - gt_normalized).abs()

        C = diff.shape[-1]

        main_loss = (sigma[..., None].expand(-1, -1, -1, C) * diff)[
            valid_mask[..., None].expand(-1, -1, -1, C)
        ].mean()

        if pred.shape[-1] == 1:
            grad_loss = self.image_gradient_loss(
                pred_aligned, gt_normalized, valid_mask
            )
        else:
            grad_loss = self.gradient_loss_multi_scale(
                pred_aligned, gt_normalized, valid_mask
            )
        reg_loss = -self.alpha * torch.log(sigma.clamp(min=1e-6))[valid_mask].mean()
        # return main + reg
        return self.gamma * main_loss + grad_loss + reg_loss


class TrackLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
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


class FinetuneLoss(MultiLoss):
    def __init__(self, lambda_track: float = 0.05) -> None:
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)

    def get_name(self) -> str:
        return "FinetuneLoss"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        # ---------- Lcamera ----------
        T = []
        for g in gts:
            T_c2w = g["camera_pose"]
            if not torch.is_tensor(T_c2w):
                T_c2w = torch.as_tensor(T_c2w)
            dtype = T_c2w.dtype
            device = T_c2w.device

            R = T_c2w[..., :3, :3]  # [...,3,3]
            t = T_c2w[..., :3, 3:4]  # [...,3,1]

            # c2w -> w2c: R^T, -R^T t
            R_w2c = R.transpose(-1, -2)  # [...,3,3]
            t_w2c = -(R_w2c @ t)  # [...,3,1]

            eye = torch.eye(4, dtype=dtype, device=device)
            T_w2c = eye.expand(*T_c2w.shape[:-2], 4, 4).clone()  # [...,4,4]
            T_w2c[..., :3, :3] = R_w2c
            T_w2c[..., :3, 3:4] = t_w2c

            if T_w2c.dim() == 2:
                T_w2c = T_w2c.unsqueeze(0)

            T.append(T_w2c)  # [B,4,4]

        T_view = torch.stack(T, dim=1)
        T_c2w_first = torch.inverse(T_view[:, 0])

        T_wprime2c = T_view @ T_c2w_first.unsqueeze(1)  # [B,V,4,4]
        camera_extrinsics_gt = T_wprime2c
        camera_intrinsics_gt = torch.stack(
            [g["camera_intrinsics"] for g in gts], dim=1
        )  # b v 3 3
        images_hw = gts[0]["img"].shape[-2:]
        cam_gt = extri_intri_to_pose_encoding(
            camera_extrinsics_gt, camera_intrinsics_gt, images_hw
        )
        cam_pr = torch.stack([p["camera_pose"] for p in preds], dim=1)

        Lcamera = self.cam_loss(cam_pr, cam_gt)

        # ---------- Ldepth ----------
        depth_terms = []
        for g, p in zip(gts, preds):
            if "depth" in p:
                sigma_p = p["depth_conf"]
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    valid_mask = torch.ones_like(g["valid_mask"])
                depth_terms.append(
                    self.depth_loss(
                        p["depth"],
                        g["depthmap"].unsqueeze(-1),
                        sigma_p=sigma_p,
                        valid_mask=valid_mask,
                    )
                )
        Ldepth = (
            torch.stack(depth_terms).mean()
            if depth_terms
            else torch.zeros_like(Lcamera)
        )

        total = Lcamera * 20 + Ldepth * 10
        details = {}

        details["Lcamera"] = float(Lcamera) * 20
        details["Ldepth"] = float(Ldepth) * 10
        details["total"] = float(total)

        return total, details


class DistillLoss(MultiLoss):
    def __init__(self, lambda_track: float = 0.05) -> None:
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)  # init 0.01 now 0.1
        self.pmap_loss = DepthOrPmapLoss(alpha=0.1)
        self.track_loss = TrackLoss()
        self.lambda_track = lambda_track

    def get_name(self) -> str:
        return "DistillLoss"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        # ---------- Lcamera ----------
        cam_gt = torch.stack([g["camera_pose"] for g in gts], dim=1)
        cam_pr = torch.stack([p["camera_pose"] for p in preds], dim=1)
        Lcamera = self.cam_loss(cam_pr, cam_gt)

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
            else torch.zeros_like(Lcamera)
        )

        # ---------- Lpmap ----------
        pmap_terms = []
        for g, p in zip(gts, preds):
            sigma_p = p["conf"]
            sigma_g = g["conf"]
            valid_mask = g["valid_mask"]
            if not valid_mask.any():
                valid_mask = torch.ones_like(g["valid_mask"])
            pmap_terms.append(
                self.pmap_loss(
                    p["pts3d_in_other_view"],
                    g["pts3d_in_other_view"],
                    sigma_p,
                    sigma_g,
                    valid_mask,
                )
            )
        Lpmap = torch.stack(pmap_terms).mean()

        # ---------- Ltrack ----------
        if ("track" in gts[0]) and ("track" in preds[0]):
            y_gt = torch.stack([g["track"] for g in gts], dim=1)
            vis_gt = torch.stack([g["vis"] for g in gts], dim=1)

            y_pr = torch.stack([p["track"] for p in preds], dim=1)
            vis_pr = torch.stack([p["vis"] for p in preds], dim=1)

            w_p = torch.stack([p["track_conf"] for p in preds], dim=1)
            w_g = torch.stack([g["track_conf"] for g in gts], dim=1)

            Ltrack = self.track_loss(y_pr, y_gt, vis_pr, vis_gt, w_p, w_g)
        else:
            Ltrack = torch.zeros_like(Lcamera)

        total = (
            Lcamera * 20 + Ldepth * 20 + Lpmap * 10 + self.lambda_track * 10 * Ltrack
        )
        details = {}

        details["Lcamera"] = float(Lcamera) * 20
        details["Ldepth"] = float(Ldepth) * 20
        details["Lpmap"] = float(Lpmap) * 10
        details["Ltrack"] = float(Ltrack) * self.lambda_track * 10
        details["total"] = float(total)

        return total, details
