import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.losses import MultiLoss
from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(
    prediction, target, mask, reduction=reduction_batch_based, frame_id_mask=None
):
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


def normalize_prediction_robust(target, mask, ms=None):
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


def compute_scale_and_shift(prediction, target, mask):
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
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.__prediction_median_scale = None
        self.__target_median_scale = None

    def forward(
        self,
        prediction,
        target,
        mask,
        pred_ms=None,
        tar_ms=None,
        num_frame_h=1,
        no_norm=False,
    ):
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

    def get_median_scale(self):
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
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
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
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
        trim=0.2,
        temp_grad_scales=4,
        temp_grad_decay=0.5,
        reduction="batch-based",
        diff_depth_th=0.05,
    ):
        super().__init__()

        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(self, prediction, target, mask):
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
        self, alpha=0.5, scales=4, trim=0.0, stable_scale=10, reduction="batch-based"
    ):
        super().__init__()
        self.spatial_loss = TrimmedProcrustesLoss(
            alpha=alpha, scales=scales, trim=trim, reduction=reduction
        )
        self.stable_loss = TemporalGradientMatchingLoss(
            trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1
        )
        self.stable_scale = stable_scale

    def forward(self, prediction, target, mask):
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


def closed_form_scale_and_shift(pred, gt):
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


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
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


def point_map_to_normal(point_map, mask, eps=1e-6):
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


def check_and_fix_inf_nan(tensor, name, hard_max=100):
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    if invalid_mask.any():
        print(
            f"[warning] {name} contains {invalid_mask.sum().item()} inf/nan values, replacing with 0"
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=hard_max, neginf=-hard_max)
    return tensor


class CameraLoss(nn.Module):
    def __init__(self, delta=1e-1, weights=(1.0, 1.0, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, pred_pose, gt_pose):
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
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.grad_scales = 3
        self.gamma = 1.0

    def gradient_loss_multi_scale(self, pred, gt, mask=None):
        total = 0
        for s in range(self.grad_scales):
            step = 2**s
            pred_s = pred[:, ::step, ::step]
            gt_s = gt[:, ::step, ::step]
            mask_s = mask[:, ::step, ::step]
            total += self.normal_loss(pred_s, gt_s, mask_s)
        return total / self.grad_scales

    def normal_loss(self, pred, gt, mask=None):
        pred_norm, _ = point_map_to_normal(pred, mask)
        gt_norm, _ = point_map_to_normal(gt, mask)
        cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=-1)
        return 1 - cos_sim.mean()

    def image_gradient_loss(self, pred, gt, mask=None):
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

    def forward(self, pred, gt, sigma_p=None, sigma_g=None, valid_mask=None):
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
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = 0.2
        self.gamma = 1.0

    def forward(self, y_pr, y_gt, vis_pr, vis_gt, w_p, w_g):
        # w = 0.5 * (w_p + w_g)
        w = w_p
        l_pos = (y_pr - y_gt).norm(dim=-1)
        l_pos = (w * l_pos).mean()

        l_vis = self.bce(vis_pr, vis_gt.float())
        l_vis = (w * l_vis).mean()
        return l_pos + l_vis


class FinetuneLoss(MultiLoss):
    def __init__(self, lambda_track=0.05):
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)

    def get_name(self):
        return "FinetuneLoss"

    def compute_loss(self, gts, preds, track_queries=None, track_preds=None):
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
    def __init__(self, lambda_track=0.05):
        super().__init__()
        self.cam_loss = CameraLoss(delta=0.1, weights=(1.0, 1.0, 0.5))
        self.depth_loss = DepthOrPmapLoss(alpha=0.1)  # init 0.01 now 0.1
        self.pmap_loss = DepthOrPmapLoss(alpha=0.1)
        self.track_loss = TrackLoss()
        self.lambda_track = lambda_track

    def get_name(self):
        return "DistillLoss"

    def compute_loss(self, gts, preds, track_queries=None, track_preds=None):
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
