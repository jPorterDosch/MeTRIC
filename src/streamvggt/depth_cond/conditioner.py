"""DepthConditioner: sparse metric depth -> features for a chosen injection point.

One module used by both injection paths. Always operates on [B, S, H, W];
S=1 is the degenerate single-frame case and is never branched on.

Extension point (NOT implemented): a future PoseConditioner (MLP-encoded
rotations/scales, a la MapAnything/POW3R) can be registered as another
conditioner producing the same output contract as the project step here:
  - InjectionType.HEAD:  {head_name: [per-scale residual [B,S,features,h_i,w_i]] | None}
  - InjectionType.TOKEN: [B, S, P_patch, token_dim]
Anything meeting that contract can be summed with this module's output at the
injection site in model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DepthCondCfg,
    EncoderType,
    HeadType,
    InjectionType,
    NormType,
    TemporalType,
)


def masked_downsample(
    disp: torch.Tensor,
    mask: torch.Tensor,
    out_hw: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked average pooling for sparse maps.

    Standard average pooling is wrong here: it blends "0 m" with "no
    measurement". This pools only over valid pixels and carries the
    fraction-valid map as the new validity channel.

    disp, mask: [N, 1, H, W]; mask in {0, 1}.
    Returns (pooled [N,1,h,w], frac [N,1,h,w]).
    """
    valid_sum = F.adaptive_avg_pool2d(disp * mask, out_hw)  # mean(disp * mask) per cell
    frac = F.adaptive_avg_pool2d(mask, out_hw)  # fraction of valid pixels per cell
    pooled = valid_sum / frac.clamp(min=eps)  # mean over valid pixels only
    pooled = pooled * (frac > 0)  # exactly 0 where a cell had no valid pixel
    return pooled, frac


class ConvDepthEncoder(nn.Module):
    """Small conv stem (2 -> C channels) that brings the sparse (disparity,
    validity) map to the backbone's patch-grid resolution (stride = patch_size),
    following MapAnything's conv depth encoder in spirit."""

    def __init__(self, channels: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stem = nn.Conv2d(2, channels, kernel_size=patch_size, stride=patch_size)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, 2, H, W] -> [B, S, C, H/ps, W/ps]
        B, S = x.shape[:2]
        y = self.stem(x.flatten(0, 1))
        y = y + self.refine(y)
        return y.reshape(B, S, *y.shape[1:])


class TemporalAttention(nn.Module):
    """Causal self-attention over the S axis only, applied independently at
    every spatial location. Residual with a zero-initialized output projection,
    so at initialization it is an exact no-op for every S (which subsumes the
    required S=1 no-op).

    The fused qkv linear is standard multi-head practice: one Linear(dim, 3*dim)
    computes q, k, v for ALL heads at once; the reshape below splits the output
    into per-head subspaces, so each head still attends independently.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = 1 if dim < 64 else 4
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W] -> same shape
        B, S, C, H, W = x.shape
        t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        h = self.norm(t)
        qkv = self.qkv(h).reshape(B * H * W, S, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [BHW, heads, S, hd]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B * H * W, S, C)
        t = t + self.out(y)
        return t.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)


class DepthConditioner(nn.Module):
    """Sparse metric depth -> features for a chosen injection point.

    Input is always [B, S, H, W] (S=1 = single frame). encoder & temporal both
    optional, selected via cfg (never hard-coded here). cfg must already be
    validated (once, at the entrypoint / MetricCfg.validate()).

    out_spec:
      - InjectionType.HEAD:  {"features": int, "num_scales": int, "heads": [HeadType]}
      - InjectionType.TOKEN: {"token_dim": int}
    """

    def __init__(self, cfg: DepthCondCfg, out_spec: dict, patch_size: int = 14) -> None:
        super().__init__()
        self.cfg = cfg
        self.injection = cfg.injection
        self.out_spec = out_spec
        self.patch_size = patch_size

        # --- AXIS 1: encoder ---
        match cfg.encoder:
            case EncoderType.IDENTITY:
                self.encoder = None  # raw passthrough ("naive")
                enc_channels = 2
            case EncoderType.CONV:
                self.encoder = ConvDepthEncoder(cfg.conv_channels, patch_size)
                enc_channels = cfg.conv_channels
            case EncoderType.MAE:
                raise NotImplementedError(
                    "depth_cond.encoder='mae' (MAE-style encoder for the sparse input) is a "
                    "planned variant that has not been built yet. Use 'identity' or 'conv'."
                )
            case _:
                raise ValueError(f"unknown encoder type: {cfg.encoder!r}")
        self._enc_channels = enc_channels

        # --- AXIS 3: temporal mixing (over S only) ---
        match cfg.temporal:
            case TemporalType.NONE:
                self.temporal = None
            case TemporalType.ATTENTION:
                self.temporal = TemporalAttention(enc_channels)
            case _:
                raise ValueError(f"unknown temporal type: {cfg.temporal!r}")

        # --- projection to the injection interface ---
        match self.injection:
            case InjectionType.HEAD:
                features = out_spec["features"]
                num_scales = out_spec["num_scales"]
                # One zero-initialized conv per (target head, fusion scale). By
                # linearity conv([x; d]) == conv(x) + conv(d), so adding these
                # residuals after the head's frozen layer{i}_rn convs is numerically
                # identical to widening those convs' in_channels with zero-init
                # slices -- while leaving the pretrained weights untouched.
                self.head_convs = nn.ModuleDict(
                    {
                        head.value: nn.ModuleList(
                            [
                                nn.Conv2d(
                                    enc_channels,
                                    features,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False,
                                )
                                for _ in range(num_scales)
                            ]
                        )
                        for head in out_spec["heads"]
                    }
                )
                for convs in self.head_convs.values():
                    for c in convs:
                        nn.init.zeros_(c.weight)
            case InjectionType.TOKEN:
                token_dim = out_spec["token_dim"]
                if cfg.encoder == EncoderType.IDENTITY:
                    # Lossless "raw passthrough": fold each patch_size x patch_size
                    # patch of both channels into the channel dim, then one linear.
                    in_ch = 2 * patch_size * patch_size
                else:
                    in_ch = enc_channels
                self.token_proj = nn.Conv2d(in_ch, token_dim, kernel_size=1)
                nn.init.normal_(self.token_proj.weight, std=0.02)
                nn.init.zeros_(self.token_proj.bias)
                # Learnable scalar gate initialized to 0: at init the injected signal
                # is a no-op and the model output equals the pretrained baseline.
                # (Only ONE zero in the path -- a zero gate on top of a zero-init
                # projection would kill both gradients.)
                # TODO(soft deadlock): one zero is still one too many when it is a
                # scalar on the WHOLE branch: grad(token_proj) is scaled by the
                # gate, so at gate=0 token_proj gets no gradient and can only
                # learn ~|gate| slower thereafter. The gate, meanwhile, only grows
                # if the (still-random) token_proj output is useful -- so it hovers
                # at ~0 (measured -0.003 after 5 epochs in the HAMMER sweep, arm
                # b536d87d) and the token arm effectively cannot bootstrap. Fix by
                # copying the LoRA pattern (zero-init the token_proj OUTPUT and
                # drop the scalar gate: its grad does not depend on itself, cf.
                # lora_B which trained fine in the same runs), or init the gate
                # nonzero (~0.1), or give the gate its own high-LR param group.
                self.gate = nn.Parameter(torch.zeros(()))
            case _:
                raise ValueError(f"unknown injection type: {self.injection!r}")

    # ------------------------------------------------------------------
    # Depth preprocessing: where the metric claim lives or dies.
    # ------------------------------------------------------------------
    def prepare(self, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B,S,H,W] depth + mask -> [B,S,2,H,W]: (transformed disparity, validity).

        Works in disparity (1/depth). NormType.FIXED divides disparity by the
        fixed physical constant 1/norm_constant_m (same constant for every
        frame and sample); NormType.RAW keeps raw metric disparity. log1p (if
        enabled) is applied AFTER the fixed scaling. All transforms are fixed
        and monotone, so absolute metric scale is preserved. Per-frame
        normalization is not representable (see NormType).

        Invalid pixels carry 0 in channel 0 (not NaN, not a sentinel); the mask
        channel is how the model distinguishes "0 signal" from "no measurement".
        """
        mask = mask.to(dtype=depth.dtype)
        disp = 1.0 / depth.clamp(min=1e-3)
        match self.cfg.norm:
            case NormType.FIXED:
                disp = disp * self.cfg.norm_constant_m  # disp / (1 / norm_constant_m)
            case NormType.RAW:
                pass  # raw metric disparity, no scaling
            case _:
                raise ValueError(f"unhandled norm type: {self.cfg.norm!r}")
        if self.cfg.log_depth:
            disp = torch.log1p(disp)
        disp = disp * mask  # a hole and a zero reading must never look the same:
        # holes are 0 in ch0 AND 0 in ch1; readings keep ch1 = 1.
        return torch.stack([disp, mask], dim=2)

    # ------------------------------------------------------------------
    def forward(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        out_hw_list: list[tuple[int, int]] | None = None,
    ) -> dict[str, list[torch.Tensor] | None] | torch.Tensor:
        """depth, mask: [B,S,H,W].

        HEAD arm: requires out_hw_list (spatial size of each DPT fusion scale);
                  returns {head_name: [per-scale [B,S,features,h_i,w_i]] | None}
                  with an entry for EVERY HeadType (None = head not conditioned).
        TOKEN arm: returns gated token features [B,S,P_patch,token_dim].
        """
        x = self.prepare(depth, mask)  # [B,S,2,H,W]
        if self.encoder is not None:
            x = self.encoder(x)
        if self.temporal is not None:
            x = self.temporal(x)

        match self.injection:
            case InjectionType.HEAD:
                if out_hw_list is None:
                    raise ValueError(
                        "head injection needs per-scale output sizes (out_hw_list); "
                        "compute them with dpt_fusion_sizes(H, W, patch_size)"
                    )
                return self._project_head(x, out_hw_list)
            case InjectionType.TOKEN:
                return self._project_token(x)
            case _:
                raise ValueError(f"unknown injection type: {self.injection!r}")

    def _project_head(
        self, x: torch.Tensor, out_hw_list: list[tuple[int, int]]
    ) -> dict[str, list[torch.Tensor] | None]:
        """Projection for InjectionType.HEAD (the post-cache CONTROL arm):
        produce one additive residual per (target DPT head, fusion scale),
        matched to that scale's spatial size.

        The per-scale input depends on the encoder: the IDENTITY path uses
        masked multi-scale pooling of the raw (disparity, validity) map; any
        learned encoder (CONV) instead has its dense latent resized, since its
        channels are no longer a (signal, mask) pair that masked pooling
        understands.
        """
        B, S = x.shape[:2]
        flat = x.flatten(0, 1)  # [B*S, C, h, w]
        # Per-scale inputs, once (shared across target heads).
        scale_inputs = []
        for out_hw in out_hw_list:
            match self.cfg.encoder:
                case EncoderType.IDENTITY:
                    # channel 0 is the (possibly temporally-mixed) signal,
                    # channel 1 the validity
                    pooled, frac = masked_downsample(flat[:, 0:1], flat[:, 1:2], out_hw)
                    scale_inputs.append(torch.cat([pooled, frac], dim=1))
                case EncoderType.CONV:
                    # learned dense latent: channels are no longer a
                    # (signal, mask) pair, so resize instead of masked-pool
                    scale_inputs.append(
                        F.interpolate(
                            flat, size=out_hw, mode="bilinear", align_corners=False
                        )
                    )
                case _:
                    # new EncoderType members must be threaded here explicitly
                    # (decide: masked pooling or latent resize?)
                    raise ValueError(
                        f"no head-arm per-scale projection defined for encoder "
                        f"{self.cfg.encoder!r}"
                    )
        out: dict[str, list[torch.Tensor] | None] = {h.value: None for h in HeadType}
        for head, convs in self.head_convs.items():
            residuals = []
            for conv, inp in zip(convs, scale_inputs):
                r = conv(inp)
                residuals.append(r.reshape(B, S, *r.shape[1:]))
            out[head] = residuals
        return out

    def _project_token(self, x: torch.Tensor) -> torch.Tensor:
        """Projection for InjectionType.TOKEN (the pre-cache PROPOSED arm):
        produce per-patch features at the backbone patch grid, gated by the
        zero-init scalar, to be added residually to the RGB patch tokens."""
        B, S = x.shape[:2]
        flat = x.flatten(0, 1)  # [B*S, C, h, w]
        match self.cfg.encoder:
            case EncoderType.IDENTITY:
                # raw full-res map: fold patches into channels to reach the patch grid
                flat = F.pixel_unshuffle(flat, self.patch_size)  # [B*S, 2*ps^2, ph, pw]
            case EncoderType.CONV:
                pass  # conv stem already produced a patch-grid latent
            case _:
                # new EncoderType members must be threaded here explicitly
                raise ValueError(
                    f"no token-arm projection input defined for encoder {self.cfg.encoder!r}"
                )
        y = self.token_proj(flat)  # [B*S, token_dim, ph, pw]
        y = y.flatten(2).transpose(1, 2)  # [B*S, P_patch, token_dim]
        y = self.gate * y
        return y.reshape(B, S, *y.shape[1:])


def dpt_fusion_sizes(H: int, W: int, patch_size: int) -> list[tuple[int, int]]:
    """Spatial size of each DPT fusion scale for an input of size (H, W),
    matching DPTHead.resize_layers: 4x, 2x, 1x, and stride-2 conv (ceil /2)
    of the (H/ps, W/ps) patch grid."""
    ph, pw = H // patch_size, W // patch_size
    return [
        (4 * ph, 4 * pw),
        (2 * ph, 2 * pw),
        (ph, pw),
        ((ph + 1) // 2, (pw + 1) // 2),
    ]
