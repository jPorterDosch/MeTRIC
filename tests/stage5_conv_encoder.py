"""Stage 5 CHECK: conv depth encoder.

Shapes flow end-to-end for both injection points; with depth zeroed the head
arm still matches baseline (the per-scale projections after the conv stem are
zero-initialized, so preservation is exact by construction even though the
stem has biases), and the token arm matches via its zero gate.
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


def make_cfg(injection):
    from streamvggt.depth_cond import MetricCfg

    cfg = MetricCfg()
    cfg.depth_cond.injection = injection
    cfg.depth_cond.encoder = "conv"
    cfg.lora.enabled = True
    cfg.train.grad_checkpoint = False
    return cfg.validate()


def main():
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

    for injection in ("head", "token"):
        model = MetricStreamVGGT(make_cfg(injection))
        model.load_pretrained(CKPT)
        model.apply_lora_adapters()
        model.freeze_for_finetune()
        model = model.to(dev).eval()

        # end-to-end with real sparse depth: shapes must flow
        out_real = collect_outputs(model, views)
        for key, t in out_real.items():
            assert t.shape == ref[key].shape, (
                f"{injection}/{key}: {t.shape} vs {ref[key].shape}"
            )
        print(f"[stage5] conv encoder + {injection}: end-to-end shapes OK")

        # zero depth must still match baseline at init
        out_zero = collect_outputs(model, zero_depth(views))
        d, k = max_abs_diff(ref, out_zero)
        print(
            f"[stage5] conv encoder + {injection}, depth zeroed: max|diff| = {d:.3e} ({k})"
        )
        assert d <= TOL, f"FAIL ({injection}): {d} > {TOL}"
        free(model)

    print("STAGE 5 PASS")


if __name__ == "__main__":
    main()
