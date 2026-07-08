"""Shared helpers for the depth-conditioning stage checks.

Run each stage script directly with the StreamVGGT conda env python, e.g.:
    /users/jdosch/miniconda3/envs/StreamVGGT/bin/python tests/stage1_head_zero_init.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CKPT = os.path.join(ROOT, "ckpt", "checkpoints.pth")

import torch  # noqa: E402


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_views(
    B=1, S=2, H=154, W=140, seed=0, dev=None, with_sparse_depth=True, n_sparse=300
):
    """Synthetic clip: random RGB in [0,1], smooth synthetic metric depth,
    sparse samples of it + validity mask."""
    g = torch.Generator().manual_seed(seed)
    dev = dev or device()
    views = []
    ys = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    for s in range(S):
        img = torch.rand(B, 3, H, W, generator=g)
        depth = 1.5 + 3.0 * ys + 0.5 * torch.rand(B, H, W, generator=g) + 0.1 * s
        view = {"img": img.to(dev)}
        if with_sparse_depth:
            mask = torch.zeros(B, H, W, dtype=torch.bool)
            for b in range(B):
                idx = torch.randperm(H * W, generator=g)[:n_sparse]
                mask[b].view(-1)[idx] = True
            view["sparse_depth"] = (depth * mask).to(dev)
            view["sparse_depth_mask"] = mask.to(dev)
            view["depthmap"] = depth.to(dev)
        views.append(view)
    return views


def zero_depth(views):
    out = []
    for v in views:
        v2 = dict(v)
        if "sparse_depth" in v2:
            v2["sparse_depth"] = torch.zeros_like(v2["sparse_depth"])
            v2["sparse_depth_mask"] = torch.zeros_like(v2["sparse_depth_mask"])
        out.append(v2)
    return out


@torch.no_grad()
def collect_outputs(model, views, query_points=None):
    """Forward and pull the comparable prediction tensors out of the output."""
    out = model(views, query_points)
    assert out.ress, (
        "model returned no per-frame outputs; equivalence check would be vacuous"
    )
    res = {}
    for s, r in enumerate(out.ress):
        res[f"depth_{s}"] = r["depth"].float().cpu()
        res[f"depth_conf_{s}"] = r["depth_conf"].float().cpu()
        res[f"pts3d_{s}"] = r["pts3d_in_other_view"].float().cpu()
        res[f"conf_{s}"] = r["conf"].float().cpu()
        res[f"pose_{s}"] = r["camera_pose"].float().cpu()
    return res


def max_abs_diff(a: dict, b: dict):
    assert set(a) == set(b), (sorted(a), sorted(b))
    assert a, "empty output dicts; comparison would be vacuous"
    worst = 0.0
    worst_key = None
    for k in a:
        # NaN guard: nan > worst is False, so without this check two all-NaN
        # outputs would compare as "identical" and the test would pass on garbage
        assert torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all(), (
            f"non-finite values in {k}; refusing a NaN-vs-NaN 'equivalence'"
        )
        d = (a[k] - b[k]).abs().max().item()
        if d > worst:
            worst, worst_key = d, k
    return worst, worst_key


def load_ckpt_cpu():
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    if (
        isinstance(sd, dict)
        and "model" in sd
        and not any(k.startswith("aggregator.") for k in sd)
    ):
        sd = sd["model"]
    return sd


def free(model):
    """Release a model's GPU memory. `del model` here only unbinds this local
    parameter -- the caller still holds its own reference, so the object cannot
    be collected. Moving it to CPU frees the GPU allocation regardless, which is
    what the stage tests actually need to avoid holding two full models on the
    device at once."""
    import gc

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
