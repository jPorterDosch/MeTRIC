"""Stage 1 CHECK (CRITICAL): head injection preserves the pretrained checkpoint.

Run pretrained StreamVGGT and the head-injected MetricStreamVGGT on the same
RGB with depth zeroed; outputs must be equal within 1e-4. Proves the zero-init
construction preserves the checkpoint exactly.
"""

import torch

from common import (
    CKPT,
    collect_outputs,
    free,
    device,
    load_ckpt_cpu,
    make_views,
    max_abs_diff,
    zero_depth,
)

TOL = 1e-4


def main():
    dev = device()
    torch.manual_seed(0)
    views = make_views(B=1, S=2, dev=dev)

    sd = load_ckpt_cpu()

    # --- baseline: unmodified pretrained StreamVGGT ---
    from streamvggt.models.streamvggt import StreamVGGT

    base = StreamVGGT()
    load_result = base.load_state_dict(sd, strict=True)
    print(f"baseline strict load: {load_result}")
    base = base.to(dev).eval()
    ref = collect_outputs(base, views)
    free(base)

    # --- head-injected model (identity encoder, control arm), LoRA off for this stage ---
    from streamvggt.depth_cond import MetricCfg, MetricStreamVGGT

    cfg = MetricCfg()
    cfg.depth_cond.injection = "head"
    cfg.depth_cond.encoder = "identity"
    cfg.lora.enabled = False
    cfg.train.grad_checkpoint = False
    cfg.validate()

    model = MetricStreamVGGT(cfg)
    model.load_pretrained(CKPT)
    model = model.to(dev).eval()

    # depth zeroed (the required check)
    out_zero = collect_outputs(model, zero_depth(views))
    d, k = max_abs_diff(ref, out_zero)
    print(
        f"[stage1] head-injected vs pretrained, depth zeroed: max|diff| = {d:.3e} ({k})"
    )
    assert d <= TOL, f"FAIL: {d} > {TOL}"

    # bonus: with REAL sparse depth the zero-init convs must still be a no-op at init
    out_real = collect_outputs(model, views)
    d2, k2 = max_abs_diff(ref, out_real)
    print(
        f"[stage1] head-injected vs pretrained, real sparse depth: max|diff| = {d2:.3e} ({k2})"
    )
    assert d2 <= TOL, f"FAIL: {d2} > {TOL}"

    print("STAGE 1 PASS")


if __name__ == "__main__":
    main()
