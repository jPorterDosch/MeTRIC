"""Stage 2 CHECK: masked multi-scale pooling.

(a) a cell with no valid pixels pools to exactly 0 with frac 0;
(b) a cell with known valid values pools to their mean (not diluted by zeros);
(c) produced spatial sizes match the real DPT head fusion scales.
"""

import torch

from common import ROOT  # noqa: F401  (sets sys.path)

from streamvggt.depth_cond import masked_downsample, dpt_fusion_sizes
from streamvggt.heads.dpt_head import DPTHead


def check_ab():
    # 4x4 map pooled to 2x2 -> each cell is a 2x2 block.
    disp = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    # top-left block: two valid pixels 3.0 and 5.0 (mean 4.0), two invalid zeros
    disp[0, 0, 0, 0], mask[0, 0, 0, 0] = 3.0, 1.0
    disp[0, 0, 1, 1], mask[0, 0, 1, 1] = 5.0, 1.0
    # top-right block: entirely invalid
    # bottom-left block: one valid pixel 2.0
    disp[0, 0, 3, 0], mask[0, 0, 3, 0] = 2.0, 1.0
    # bottom-right block: all four valid, values 1..4 (mean 2.5)
    disp[0, 0, 2:, 2:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask[0, 0, 2:, 2:] = 1.0

    pooled, frac = masked_downsample(disp, mask, (2, 2))

    assert pooled[0, 0, 0, 1] == 0.0 and frac[0, 0, 0, 1] == 0.0, (
        "empty cell must pool to exactly 0 / frac 0"
    )
    assert torch.allclose(pooled[0, 0, 0, 0], torch.tensor(4.0), atol=1e-5), (
        f"masked mean diluted by zeros: got {pooled[0, 0, 0, 0].item()} want 4.0"
    )
    assert torch.allclose(frac[0, 0, 0, 0], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(pooled[0, 0, 1, 0], torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(frac[0, 0, 1, 0], torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(pooled[0, 0, 1, 1], torch.tensor(2.5), atol=1e-5)
    assert torch.allclose(frac[0, 0, 1, 1], torch.tensor(1.0), atol=1e-6)
    print("[stage2] (a)+(b) masked pooling semantics: OK")


def check_c():
    # Discover the true fusion-scale sizes by pushing tokens through a real
    # (randomly initialized) DPTHead's project+resize pipeline, and compare
    # with dpt_fusion_sizes (which the conditioner uses).
    H, W, ps = 154, 140, 14
    ph, pw = H // ps, W // ps
    head = DPTHead(
        dim_in=64, out_channels=[16, 32, 48, 48], features=8, pos_embed=False
    )
    x = torch.randn(1, 64, ph, pw)
    true_sizes = []
    for i in range(4):
        y = head.resize_layers[i](head.projects[i](x))
        true_sizes.append(tuple(y.shape[-2:]))
    pred_sizes = [tuple(s) for s in dpt_fusion_sizes(H, W, ps)]
    assert pred_sizes == true_sizes, f"{pred_sizes} != {true_sizes}"

    # and the conditioner's head-arm outputs actually match those sizes
    from streamvggt.depth_cond import (
        DepthCondCfg,
        DepthConditioner,
        EncoderType,
        HeadType,
        InjectionType,
    )

    cfg = DepthCondCfg(injection=InjectionType.HEAD, encoder=EncoderType.IDENTITY)
    cfg.validate()
    cond = DepthConditioner(
        cfg, {"features": 8, "num_scales": 4, "heads": [HeadType.DEPTH]}, patch_size=ps
    )
    depth = torch.rand(1, 2, H, W) + 0.5
    mask = (torch.rand(1, 2, H, W) > 0.9).float()
    residuals = cond(depth, mask, out_hw_list=pred_sizes)["depth"]
    got = [tuple(r.shape[-2:]) for r in residuals]
    assert got == true_sizes, f"{got} != {true_sizes}"
    print(f"[stage2] (c) fusion-scale sizes match DPT head: {true_sizes}")


def main():
    check_ab()
    check_c()
    print("STAGE 2 PASS")


if __name__ == "__main__":
    main()
