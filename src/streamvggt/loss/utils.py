import torch
import torch.nn.functional as F

from typing import Callable


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
    _ = pred.device

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
