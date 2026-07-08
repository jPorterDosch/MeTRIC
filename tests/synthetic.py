"""Synthetic-clip helpers for the stage checks (moved out of the training
entrypoint: test scaffolding does not belong in finetune_depth.py)."""

import torch
from torch.utils.data import DataLoader, Dataset

from common import ROOT  # noqa: F401  (sets sys.path)

from streamvggt.loss.loss import *  # noqa: F401,F403  needed to eval() criterion strings


def make_synthetic_clip(
    num_views: int,
    B: int = 1,
    H: int = 154,
    W: int = 140,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> list[dict]:
    """A geometrically consistent synthetic clip: smooth RGB, metric depth,
    identity poses, pinhole intrinsics, pts3d by unprojection. Views carry
    every GT field the repo's training criterion needs."""
    g = torch.Generator().manual_seed(seed)
    f = 0.8 * max(H, W)
    K = torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])
    us = torch.arange(W).float().unsqueeze(0).expand(H, W)
    vs = torch.arange(H).float().unsqueeze(1).expand(H, W)
    ys = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    views = []
    for s in range(num_views):
        img = torch.zeros(B, 3, H, W)
        img[:, 0] = ys
        img[:, 1] = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
        img[:, 2] = 0.5
        img = (img + 0.1 * torch.rand(B, 3, H, W, generator=g)).clamp(0, 1)
        depth = 1.5 + 3.0 * ys + 0.2 * torch.rand(B, H, W, generator=g) + 0.05 * s
        z = depth
        pts3d = torch.stack(
            [(us - W / 2) / f * z, (vs - H / 2) / f * z, z], dim=-1
        )  # [B,H,W,3], camera frame == world frame (identity pose)
        view = {
            "img": img,  # [0,1], as the train loop feeds the model
            "depthmap": depth,
            "pts3d": pts3d,
            "valid_mask": torch.ones(B, H, W, dtype=torch.bool),
            "sky_mask": torch.zeros(B, H, W, dtype=torch.bool),
            "camera_pose": torch.eye(4).unsqueeze(0).expand(B, 4, 4).contiguous(),
            "camera_intrinsics": K.unsqueeze(0).expand(B, 3, 3).contiguous(),
            "camera_only": torch.zeros(B, dtype=torch.bool),
            "is_metric": torch.ones(B, dtype=torch.bool),
            "is_metric_scale": torch.ones(B, dtype=torch.bool),
        }
        views.append({k: v.to(device) for k, v in view.items()})
    return views


def overfit_steps(
    model: torch.nn.Module,
    batch: list[dict],
    criterion_str: str,
    steps: int = 5,
    lr: float = 1e-4,
) -> list[float]:
    """Run a few optimization steps on one clip with the repo's real criterion;
    returns the per-step losses."""
    from dust3r.inference import loss_of_one_batch

    device = next(model.parameters()).device
    criterion = eval(criterion_str).to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95))

    losses = []
    for _ in range(steps):
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator=None,
            symmetrize_batch=False,
            use_amp=True,
        )
        loss, _ = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss))
    return losses


class _SyntheticClips(Dataset):
    """Yields synthetic clips in the dust3r view-dict format the trainer
    consumes. Images are emitted in [-1, 1] (the dataset convention) so the
    train loop's (img + 1) / 2 rescale recovers [0, 1] before the model."""

    def __init__(self, num_views: int, n_steps: int, H: int, W: int, seed: int) -> None:
        self.num_views, self.n_steps, self.H, self.W, self.seed = num_views, n_steps, H, W, seed

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> list[dict]:
        views = make_synthetic_clip(
            self.num_views, B=1, H=self.H, W=self.W, device="cpu", seed=self.seed + idx
        )
        for v in views:
            v["img"] = v["img"] * 2 - 1
        return views


def synthetic_loader(
    num_views: int = 3, n_steps: int = 12, H: int = 70, W: int = 70, seed: int = 0
) -> DataLoader:
    """DataLoader over synthetic clips. batch_size=1 with a pass-through collate
    so each iteration yields one clip (a list of view dicts), exactly the shape
    train_one_epoch/loss_of_one_batch expect; accelerate.prepare moves the
    nested tensors to the device."""
    ds = _SyntheticClips(num_views, n_steps, H, W, seed)
    return DataLoader(ds, batch_size=1, num_workers=0, collate_fn=lambda b: b[0])
