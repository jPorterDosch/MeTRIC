"""Stage 4 CHECK: frozen-encoder feature cache.

Live-encoder forward vs cached forward on the same input must agree within
1e-5. Covers both the cold path (compute + store) and the warm path (load).
"""

import os
import tempfile

import torch

from common import CKPT, collect_outputs, device, make_views, max_abs_diff

TOL = 1e-5


def main():
    dev = device()
    torch.manual_seed(0)

    from streamvggt.depth_cond import MetricCfg, MetricStreamVGGT

    with tempfile.TemporaryDirectory(prefix="enc_cache_") as cache_dir:
        cfg = MetricCfg()
        cfg.depth_cond.injection = "token"
        cfg.lora.enabled = True
        cfg.train.grad_checkpoint = False
        cfg.encoder_cache.enabled = True
        cfg.encoder_cache.dir = cache_dir
        cfg.validate()

        model = MetricStreamVGGT(cfg)
        model.load_pretrained(CKPT)
        model.apply_lora_adapters()
        model.freeze_for_finetune()
        model = model.to(dev).eval()

        views = make_views(B=1, S=2, dev=dev)

        # live path: no cache keys -> falls back to running the encoder
        ref = collect_outputs(model, views)

        # cold cached path: keys present, cache empty -> compute + store
        for s, v in enumerate(views):
            v["cache_key"] = f"synthetic_clip0_frame{s}_154x140"
        cold = collect_outputs(model, views)
        n_files = len(os.listdir(cache_dir))
        assert n_files == len(views), (
            f"expected {len(views)} cache files, found {n_files}"
        )
        d, k = max_abs_diff(ref, cold)
        print(f"[stage4] live vs cache-cold: max|diff| = {d:.3e} ({k})")
        assert d <= TOL, f"FAIL: {d} > {TOL}"

        # warm cached path: loaded from disk
        warm = collect_outputs(model, views)
        d2, k2 = max_abs_diff(ref, warm)
        print(f"[stage4] live vs cache-warm: max|diff| = {d2:.3e} ({k2})")
        assert d2 <= TOL, f"FAIL: {d2} > {TOL}"

        # positive hit check: the warm-path equivalence above is only meaningful
        # if the warm pass actually READS the stored features. Poison one cached
        # file and confirm the output changes -- proving a real cache hit rather
        # than a silent recompute-and-restore (which would also pass d2 <= TOL).
        key = views[0]["cache_key"]
        stored = model.cache.load(key)
        model.cache.save(key, stored + 1.0)
        poisoned = collect_outputs(model, views)
        d3, _ = max_abs_diff(ref, poisoned)
        print(f"[stage4] poisoned-cache hit check: max|diff| = {d3:.3e} (expect > 0)")
        assert d3 > TOL, (
            "warm pass ignored the stored file -> cache is not actually hit"
        )

    print("STAGE 4 PASS")


if __name__ == "__main__":
    main()
