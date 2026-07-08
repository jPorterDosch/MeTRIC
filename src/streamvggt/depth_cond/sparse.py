"""Simulate sparse metric depth from dense GT depthmaps via patch masking.

Real deployments feed sparse metric depth from robot-mounted sensors; for
training on RGB-D datasets we reveal a random subset of patches of the GT
depth (MAE-style), controlled by SparseSimMode:

  NONE      -- no masking: the full dense GT depth is passed as conditioning.
  RANDOM    -- random visible patches, resampled independently per frame.
  TUBE_MASK -- one patch mask sampled per clip and shared by every frame
               (a "tube" through time, matching video-MAE terminology).

This is training wiring, not dataset generation: it runs on already-loaded
batches. Determinism is controlled by the global torch seed.
"""

import torch

from .config import SparseSimMode


def _patch_mask(
    B: int, H: int, W: int, patch_size: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Random per-sample patch visibility mask [B,H,W] (True = visible).
    Patches are patch_size x patch_size cells; ceil(num_patches*(1-ratio))
    patches are kept visible per sample."""
    gh = (H + patch_size - 1) // patch_size
    gw = (W + patch_size - 1) // patch_size
    n_patches = gh * gw
    n_visible = max(1, round(n_patches * (1.0 - mask_ratio)))
    scores = torch.rand(B, n_patches, device=device)
    keep = scores.argsort(dim=1)[:, :n_visible]
    grid = torch.zeros(B, n_patches, dtype=torch.bool, device=device)
    grid.scatter_(1, keep, True)
    grid = grid.reshape(B, gh, gw)
    mask = grid.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    return mask[:, :H, :W]


def simulate_sparse_depth(
    views: list[dict],
    mode: SparseSimMode,
    patch_size: int,
    mask_ratio: float,
) -> list[dict]:
    """For each view dict with a dense 'depthmap' [B,H,W] and no 'sparse_depth',
    add in place:
      view['sparse_depth']      [B,H,W]  (0 where masked or GT-invalid)
      view['sparse_depth_mask'] [B,H,W]  bool
    Only pixels that are GT-valid AND patch-visible count as measurements."""
    mode = SparseSimMode(mode)
    tube_mask: torch.Tensor | None = None
    for view in views:
        if "sparse_depth" in view or "depthmap" not in view:
            continue
        depth = view["depthmap"]
        if depth.dim() == 4:  # [B,H,W,1]
            depth = depth[..., 0]
        B, H, W = depth.shape
        valid = depth > 0
        if "valid_mask" in view:
            valid = valid & view["valid_mask"].to(dtype=torch.bool, device=depth.device)

        match mode:
            case SparseSimMode.NONE:
                visible = torch.ones_like(valid)
            case SparseSimMode.RANDOM:
                visible = _patch_mask(B, H, W, patch_size, mask_ratio, depth.device)
            case SparseSimMode.TUBE_MASK:
                if tube_mask is None or tube_mask.shape != valid.shape:
                    tube_mask = _patch_mask(
                        B, H, W, patch_size, mask_ratio, depth.device
                    )
                visible = tube_mask
            case _:
                raise ValueError(f"unknown sparse simulation mode: {mode!r}")

        mask = valid & visible
        view["sparse_depth"] = depth * mask
        view["sparse_depth_mask"] = mask
    return views
