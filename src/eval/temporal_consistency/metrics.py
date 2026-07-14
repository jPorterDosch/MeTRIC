from functools import lru_cache

import numpy as np
import torch

"""
This file contains evaluation metrics for the validation passes of the model.
Adapted from StreamVGGT for this use case (AbsRel, delta metrics), TAE is taken
from DepthAnyVideo.
"""


def abs_rel(gt: np.ndarray, pred: np.ndarray) -> float:
    abs_rel = (np.abs(gt - pred) / gt).mean()
    return abs_rel


@lru_cache(maxsize=8)
def _pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    # Integer pixel coordinates: this matches the unprojection convention the
    # dataset intrinsics assume (np.arange grids in streamvggt/utils/geometry
    # and dust3r/utils/geometry). Half-pixel centers put reprojected
    # coordinates exactly at .5, where np.round (half-to-even) misregisters
    # ~half the pixels, giving TAE a false noise floor for perfect static
    # predictions.
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    return ys, xs


def depth2point(
    depth: np.ndarray, mask: np.ndarray, img2lidar: np.ndarray
) -> np.ndarray:
    h, w = depth.shape
    ys, xs = _pixel_grid(h, w)
    # (H, W, 4)
    points = np.stack([xs, ys, depth, np.ones_like(xs)], axis=-1)
    points = points[mask]
    points[..., :2] *= points[..., 2:3]
    points = points @ img2lidar.T
    points = points[..., :3]
    return points


def point2depth(
    points: np.ndarray, warp_mask: np.ndarray, warp_img2lidar: np.ndarray
) -> np.ndarray:
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    lidar2img = np.linalg.inv(warp_img2lidar)
    points = points @ lidar2img.T
    depth = points[..., 2]
    eps = 1e-6
    mask = depth > eps
    cam_points = points[..., :2] / np.clip(points[..., 2:3], a_min=eps, a_max=None)
    cam_coords = np.round(cam_points).astype(np.int32)
    h, w = warp_mask.shape
    mask &= (
        (cam_coords[..., 0] >= 0)
        & (cam_coords[..., 0] < w)
        & (cam_coords[..., 1] >= 0)
        & (cam_coords[..., 1] < h)
    )
    depth = depth[mask]
    cam_coords = cam_coords[mask]
    warp_depth = np.zeros((h, w), dtype=np.float32)
    warp_depth[cam_coords[..., 1], cam_coords[..., 0]] = depth
    warp_depth = warp_depth * warp_mask
    return warp_depth


def closed_form_scale_and_shift(
    predicted_depth: torch.Tensor, ground_truth_depth: torch.Tensor
) -> tuple[float, float]:
    """Exact least-squares (s, t) minimizing ||s * pred + t - gt||^2 over the
    already-masked 1-D tensors, via the 2x2 normal equations (a few BLAS
    reductions; float64 for the accumulations). Replaces a 1000-step Adam L1
    alignment that cost ~60 s per clip on CPU and returned non-converged s/t."""
    p = predicted_depth.reshape(-1).double()
    g = ground_truth_depth.reshape(-1).double()
    n = float(p.numel())
    sp, sg = p.sum(), g.sum()
    spp, spg = p @ p, p @ g
    det = spp * n - sp * sp
    if det.abs() < 1e-12:  # constant prediction: scale is unidentifiable
        return 1.0, ((sg - sp) / n).item()
    s = (spg * n - sp * sg) / det
    t = (spp * sg - sp * spg) / det
    return s.item(), t.item()


def tae(
    depth_pred_a: np.ndarray,
    mask_a: np.ndarray,
    img2lidar_a: np.ndarray,
    depth_pred_b: np.ndarray,
    mask_b: np.ndarray,
    img2lidar_b: np.ndarray,
) -> float:
    depth_a2b = point2depth(
        depth2point(depth_pred_a, mask_a, img2lidar_a), mask_b, img2lidar_b
    )
    mask = (depth_a2b > 1e-6) & mask_b
    error_a2b = abs_rel(depth_pred_b[mask], depth_a2b[mask])
    depth_b2a = point2depth(
        depth2point(depth_pred_b, mask_b, img2lidar_b), mask_a, img2lidar_a
    )
    mask = (depth_b2a > 1e-6) & mask_a
    error_b2a = abs_rel(depth_pred_a[mask], depth_b2a[mask])
    return 0.5 * (error_a2b + error_b2a)


def depth2disparity(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(depth, torch.Tensor):
        disparity = np.zeros_like(depth.detach().cpu().numpy())
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negative_mask = depth > 0
    disparity[non_negative_mask] = 1.0 / depth[non_negative_mask]

    return disparity, non_negative_mask


def depth_evaluation(
    predicted_depth_original: np.ndarray,
    ground_truth_depth_original: np.ndarray,
    max_depth: int = 80,
    custom_mask: np.ndarray | None = None,
    post_clip_min: int | None = None,
    post_clip_max: int | None = None,
    pre_clip_min: int | None = None,
    pre_clip_max: int | None = None,
    metric_scale: bool = False,
    scale_and_shift: bool = False,
    scale_only: bool = False,
    use_gpu: bool = False,
    disp_input: bool = False,
) -> tuple[dict[str, float], torch.Tensor]:
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    # validate the mode up front: without this, the no-flag default used to
    # run median scaling, compute every metric, and only then die in the
    # parity-map dispatch below
    if not (metric_scale or scale_and_shift or scale_only):
        raise ValueError(
            "depth_evaluation requires an alignment mode: pass one of "
            "metric_scale, scale_and_shift, or scale_only"
        )

    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif scale_and_shift:
        s, t = closed_form_scale_and_shift(predicted_depth, ground_truth_depth)
        predicted_depth = s * predicted_depth + t
    elif scale_only:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = (
                residuals.abs() + 1e-8
            )  # Add small constant to avoid division by zero

            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals

            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(
                weights * predicted_depth * ground_truth_depth
            )
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)

        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()

        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth, _ = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        if custom_mask.shape != ground_truth_depth_original.shape:
            raise ValueError(
                f"{custom_mask.shape=}, expected: {ground_truth_depth_original.shape}"
            )

        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    ).item()
    sq_rel = torch.mean(
        ((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth
    ).item()

    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    # Compute the depth error parity map
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original, _ = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif scale_and_shift:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original, _ = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif scale_only:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original, _ = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    else:
        raise ValueError(
            "Only supported modes are ``metric``, ``scale_and_shift``, and ``scale_only``."
        )

    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(
        mask, depth_error_parity_map, depth_error_parity_map_full
    )

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(
        mask, ground_truth_depth_original, gt_depth_map_full
    )

    num_valid_pixels = (
        torch.sum(mask).item()
        if custom_mask is None
        else torch.sum(mask_within_mask).item()
    )
    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    return (
        results,
        depth_error_parity_map_full,
        predict_depth_map_full,
        gt_depth_map_full,
    )
