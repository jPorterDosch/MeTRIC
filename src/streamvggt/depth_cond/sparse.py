"""Simulate sparse metric depth from dense GT depthmaps.

Real deployments feed sparse metric depth from robot-mounted sensors; for
training on RGB-D datasets we subsample the GT depth. This is training wiring,
not dataset generation: it runs on already-loaded batches.
"""

import torch


def simulate_sparse_depth(views: list[dict], num_points: int) -> list[dict]:
    """For each view dict with a dense 'depthmap' [B,H,W] and no 'sparse_depth',
    sample up to num_points valid pixels per sample. Adds in place:
      view['sparse_depth']      [B,H,W]  (0 where not sampled)
      view['sparse_depth_mask'] [B,H,W]  bool
    Determinism is controlled by the global torch seed."""
    for view in views:
        if "sparse_depth" in view or "depthmap" not in view:
            continue
        depth = view["depthmap"]
        if depth.dim() == 4:  # [B,H,W,1]
            depth = depth[..., 0]
        valid = depth > 0
        if "valid_mask" in view:
            valid = valid & view["valid_mask"].to(dtype=torch.bool, device=depth.device)
        B = depth.shape[0]
        mask = torch.zeros_like(depth, dtype=torch.bool)
        flat_mask = mask.view(B, -1)
        for b in range(B):
            idx = valid[b].reshape(-1).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            sel = idx[torch.randperm(idx.numel(), device=idx.device)[:num_points]]
            flat_mask[b, sel] = True
        view["sparse_depth"] = depth * mask
        view["sparse_depth_mask"] = mask
    return views
