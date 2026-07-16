"""Stage 7 CHECK: training wiring.

Both arms are built through the entrypoint's build_model with ONLY
depth_cond.injection changed, then overfit a synthetic clip for 5 steps with
the repo's real criterion:
  * losses decrease monotonically-ish in both arms;
  * grad checkpointing is on;
  * the experiment hash separates the two arms (different dirs) and the
    entrypoint fails fast on a directory collision with a completed run.
"""

import os
import sys
import tempfile

import torch

from common import CKPT, device, free
from synthetic import make_synthetic_clip, overfit_steps

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from streamvggt.depth_cond import (
    InjectionType,
    MetricCfg,
    experiment_hash,
    experiment_id,
    simulate_sparse_depth,
)
from finetune_depth import (
    FinetuneDepthCfg,
    build_manifest,
    build_model,
    resolve_output_dir,
)


def make_cfg(injection: InjectionType) -> FinetuneDepthCfg:
    cfg = FinetuneDepthCfg(pretrained=CKPT)
    cfg.depth_cond.injection = injection
    return cfg


def check_overfit() -> None:
    dev = device()
    for injection in (InjectionType.HEAD, InjectionType.TOKEN):
        torch.manual_seed(0)
        cfg = make_cfg(injection)
        mcfg = MetricCfg(
            depth_cond=cfg.depth_cond,
            lora=cfg.lora,
            encoder_cache=cfg.encoder_cache,
            train=cfg.train,
        ).validate()
        model, _ = build_model(cfg, mcfg, dev)
        assert model.model.aggregator.grad_checkpointing, (
            "grad checkpointing must be on"
        )

        batch = make_synthetic_clip(4, device=dev, seed=0)
        simulate_sparse_depth(
            batch,
            mode=mcfg.depth_cond.sim_mode,
            patch_size=mcfg.depth_cond.sim_patch_size,
            mask_ratio=mcfg.depth_cond.sim_mask_ratio,
        )
        losses = overfit_steps(model, batch, cfg.loss, steps=5)
        rises = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        print(
            f"[stage7] {injection.value}: {losses[0]:.4f} -> {losses[-1]:.4f} (rises {rises})"
        )
        assert losses[-1] < losses[0], (
            f"{injection.value}: loss did not decrease: {losses}"
        )
        assert rises <= 1, f"{injection.value}: loss not monotonically-ish: {losses}"
        free(model)


def check_hash_and_collision() -> None:
    head_cfg, token_cfg = make_cfg(InjectionType.HEAD), make_cfg(InjectionType.TOKEN)
    mh = build_manifest(head_cfg)
    mt = build_manifest(token_cfg)
    assert experiment_hash(mh) != experiment_hash(mt), (
        "arms must land in different dirs"
    )
    # sanity: the two manifests differ in injection and nothing else
    diff = {k for k in mh if mh[k] != mt[k]}
    assert diff == {"depth_cond.injection"}, f"unexpected manifest diffs: {diff}"
    # identical config -> identical hash (stable identity)
    assert experiment_hash(mh) == experiment_hash(
        build_manifest(make_cfg(InjectionType.HEAD))
    )
    print("[stage7] experiment hashes: stable, and arms separate")

    # collision fail-fast: ANY pre-existing output dir must refuse to relaunch.
    # The dir name is built from experiment_id (the single source of truth for
    # the short-id length) -- no hardcoded slice that could desync from prod.
    with tempfile.TemporaryDirectory(prefix="stage7_dirs_") as tmp:
        cfg = make_cfg(InjectionType.HEAD)
        cfg.save_dir = tmp
        rid = experiment_id(build_manifest(cfg))
        out = os.path.join(tmp, cfg.exp_group, rid)
        os.makedirs(out)  # merely created, no checkpoints inside
        try:
            resolve_output_dir(cfg, rid)
        except RuntimeError as e:
            print(f"[stage7] collision fail-fast fired: {str(e)[:70]}...")
        else:
            raise AssertionError("resolve_output_dir did not fail on an existing dir")
        # explicit --resume bypasses the collision check
        cfg.resume = os.path.join(out, "checkpoint-last.pth")
        assert resolve_output_dir(cfg, rid) == out


def main() -> None:
    check_hash_and_collision()
    check_overfit()
    print("STAGE 7 PASS")


if __name__ == "__main__":
    main()
