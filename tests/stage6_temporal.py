"""Stage 6 CHECK: temporal attention over the S axis.

(1) At S=1, enabling temporal: attention leaves the conditioner output
    unchanged vs temporal: none within 1e-5 (holds exactly at init thanks to
    the zero-init residual output projection; see IMPLEMENTATION_NOTES.md for
    why strict post-training S=1 passthrough is not achievable).
(2) Causality: with randomized temporal weights, frame 0 of an S=3 clip equals
    the S=1 result -- earlier frames are unaffected by later ones.
(3) At init the no-op also holds for S>1 (zero-init residual).
"""

import torch

from common import ROOT  # noqa: F401  (sets sys.path)

from streamvggt.depth_cond import (
    DepthCondCfg,
    DepthConditioner,
    EncoderType,
    InjectionType,
)

TOL = 1e-5


def build(temporal, encoder=EncoderType.CONV):
    cfg = DepthCondCfg(
        injection=InjectionType.TOKEN, encoder=encoder, temporal=temporal
    )
    cfg.validate()
    return DepthConditioner(cfg, {"token_dim": 64}, patch_size=14)


def main():
    torch.manual_seed(0)
    H, W = 56, 56
    depth1 = torch.rand(2, 1, H, W) * 5 + 0.5
    mask1 = (torch.rand(2, 1, H, W) > 0.8).float()

    cond_attn = build("attention")
    cond_none = build("none")
    # share every non-temporal parameter
    cond_none.encoder.load_state_dict(cond_attn.encoder.state_dict())
    cond_none.token_proj.load_state_dict(cond_attn.token_proj.state_dict())
    with torch.no_grad():
        # open the gates so the comparison isn't trivially 0-vs-0
        cond_attn.gate.fill_(1.0)
        cond_none.gate.fill_(1.0)

    with torch.no_grad():
        # (1) S=1 no-op vs temporal: none
        out_a = cond_attn(depth1, mask1)
        out_n = cond_none(depth1, mask1)
        d = (out_a - out_n).abs().max().item()
        print(f"[stage6] S=1, attention vs none: max|diff| = {d:.3e}")
        assert d <= TOL, f"FAIL: {d} > {TOL}"

        # (3) at init the zero-init residual also makes S=3 a no-op
        depth3 = torch.rand(2, 3, H, W) * 5 + 0.5
        mask3 = (torch.rand(2, 3, H, W) > 0.8).float()
        d3 = (cond_attn(depth3, mask3) - cond_none(depth3, mask3)).abs().max().item()
        print(f"[stage6] S=3 at init, attention vs none: max|diff| = {d3:.3e}")
        assert d3 <= TOL

        # (2) causality with non-trivial temporal weights
        for p in cond_attn.temporal.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
        full = cond_attn(depth3, mask3)
        first = cond_attn(depth3[:, :1], mask3[:, :1])
        dc = (full[:, 0] - first[:, 0]).abs().max().item()
        print(f"[stage6] causality (frame0 of S=3 == S=1 run): max|diff| = {dc:.3e}")
        assert dc <= TOL, f"FAIL: temporal attention leaks future frames ({dc})"
        # sanity: later frames DO mix (otherwise the module is dead)
        assert not torch.allclose(
            full[:, 1:], cond_none(depth3, mask3)[:, 1:], atol=1e-6
        )

    print("STAGE 6 PASS")


if __name__ == "__main__":
    main()
