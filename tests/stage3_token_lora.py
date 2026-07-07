"""Stage 3 CHECKS: token injection + LoRA.

A (CRITICAL): token arm with gate at zero-init and LoRA adapters at zero-init
   (B=0), depth zeroed -> output equals the pretrained baseline within 1e-4.
B: base attention matrices requires_grad=False; trainable-param count logged
   and a small fraction of total.
C: the confound-rule assertion fires when two run manifests differ in anything
   besides depth_cond.injection.
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


def make_cfg(injection="token"):
    from streamvggt.depth_cond import MetricCfg

    cfg = MetricCfg()
    cfg.depth_cond.injection = injection
    cfg.depth_cond.encoder = "identity"
    cfg.lora.enabled = True
    cfg.train.grad_checkpoint = False
    return cfg.validate()


def check_a_and_b():
    dev = device()
    torch.manual_seed(0)
    views = make_views(B=1, S=2, dev=dev)

    sd = load_ckpt_cpu()
    from streamvggt.models.streamvggt import StreamVGGT

    base = StreamVGGT()
    base.load_state_dict(sd, strict=True)
    base = base.to(dev).eval()
    ref = collect_outputs(base, views)
    free(base)

    from streamvggt.depth_cond import MetricStreamVGGT

    model = MetricStreamVGGT(make_cfg("token"))
    model.load_pretrained(CKPT)
    n_wrapped = model.apply_lora_adapters()
    stats = model.freeze_for_finetune()
    model = model.to(dev).eval()

    # --- Check A ---
    out_zero = collect_outputs(model, zero_depth(views))
    d, k = max_abs_diff(ref, out_zero)
    print(
        f"[stage3-A] token+LoRA vs pretrained, depth zeroed: max|diff| = {d:.3e} ({k})"
    )
    assert d <= TOL, f"FAIL: {d} > {TOL}"
    # gate=0 makes even real depth a no-op at init
    out_real = collect_outputs(model, views)
    d2, k2 = max_abs_diff(ref, out_real)
    print(
        f"[stage3-A] token+LoRA vs pretrained, real sparse depth: max|diff| = {d2:.3e} ({k2})"
    )
    assert d2 <= TOL, f"FAIL: {d2} > {TOL}"

    # --- Check B ---
    assert stats["base_attention_frozen"], (
        "a base attention matrix has requires_grad=True"
    )
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    n_cond = sum(p.numel() for p in model.conditioner.parameters())
    print(f"[stage3-B] wrapped attention modules: {n_wrapped}")
    print(f"[stage3-B] total params:      {stats['total_params']:,}")
    print(
        f"[stage3-B] trainable params:  {stats['trainable_params']:,} ({stats['trainable_pct']:.3f}%)"
    )
    print(f"[stage3-B]   of which LoRA:        {n_lora:,}")
    print(f"[stage3-B]   of which conditioner: {n_cond:,}")
    assert stats["trainable_pct"] < 5.0, "trainable fraction should be small"
    assert stats["trainable_params"] == n_lora + n_cond

    # head-arm param counts for the notes (same LoRA block per the confound rule)
    free(model)
    head_model = MetricStreamVGGT(make_cfg("head"))
    head_model.apply_lora_adapters()
    head_stats = head_model.freeze_for_finetune()
    n_cond_h = sum(p.numel() for p in head_model.conditioner.parameters())
    print(
        f"[stage3-B] head arm: trainable {head_stats['trainable_params']:,} "
        f"({head_stats['trainable_pct']:.3f}%), conditioner {n_cond_h:,}"
    )
    free(head_model)


def check_c():
    from streamvggt.depth_cond import (
        ConfoundError,
        assert_confound_rule,
        experiment_manifest,
        manifest_comparable_hash,
    )

    head_cfg = make_cfg("head")
    token_cfg = make_cfg("token")

    # differing ONLY in injection: must pass, hashes must match
    assert_confound_rule(experiment_manifest(head_cfg), experiment_manifest(token_cfg))
    assert manifest_comparable_hash(head_cfg) == manifest_comparable_hash(token_cfg)
    print("[stage3-C] head vs token (only injection differs): comparable, hashes equal")

    # differing in the lora block too: must fail loudly
    bad_cfg = make_cfg("token")
    bad_cfg.lora.rank = 8
    try:
        assert_confound_rule(
            experiment_manifest(head_cfg), experiment_manifest(bad_cfg)
        )
    except ConfoundError as e:
        print(
            f"[stage3-C] confound assertion fired as required:\n    {str(e).splitlines()[0]}"
        )
    else:
        raise AssertionError(
            "confound-rule assertion did NOT fire for differing lora.rank"
        )
    assert manifest_comparable_hash(head_cfg) != manifest_comparable_hash(bad_cfg)


def main():
    check_a_and_b()
    check_c()
    print("STAGE 3 PASS")


if __name__ == "__main__":
    main()
