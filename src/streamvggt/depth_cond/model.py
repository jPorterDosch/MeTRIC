"""MetricStreamVGGT: StreamVGGT + config-driven depth conditioning.

Builds the DepthConditioner, routes its output to the configured injection
point (head = DPT fusion, post-KV-cache CONTROL; token = encoder tokens,
pre-KV-cache PROPOSED), applies LoRA to the aggregator attention, and handles
the frozen-encoder feature cache. Nothing here branches on values outside the
MetricCfg object.
"""

from typing import Optional

import torch
import torch.nn as nn

from streamvggt.models.streamvggt import StreamVGGT

from .cache import EncoderFeatureCache
from .conditioner import DepthConditioner, dpt_fusion_sizes
from .config import MetricCfg
from .lora import apply_lora, param_stats


class MetricStreamVGGT(nn.Module):
    def __init__(self, cfg: MetricCfg, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.patch_size = patch_size
        self.model = StreamVGGT(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )

        self.conditioner = None
        if cfg.depth_cond.enabled:
            if cfg.depth_cond.injection == "head":
                # Read the head geometry from the model, not from constants.
                ref_head = self.model.depth_head
                out_spec = {
                    "features": ref_head.scratch.layer1_rn.out_channels,
                    "num_scales": len(ref_head.intermediate_layer_idx),
                    "heads": list(cfg.depth_cond.heads),
                }
            else:
                out_spec = {"token_dim": embed_dim}
            self.conditioner = DepthConditioner(
                cfg.depth_cond, out_spec, patch_size=patch_size
            )

        self.cache = (
            EncoderFeatureCache(cfg.encoder_cache.dir)
            if cfg.encoder_cache.enabled
            else None
        )
        self.model.aggregator.grad_checkpointing = cfg.train.grad_checkpoint
        self._lora_applied = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def load_pretrained(self, path: str, map_location="cpu"):
        """Load the pretrained StreamVGGT checkpoint (raw state_dict) into the
        base model. Must run BEFORE apply_lora_adapters (wrapping renames keys)."""
        if self._lora_applied:
            raise RuntimeError(
                "load_pretrained must be called before apply_lora_adapters"
            )
        sd = torch.load(path, map_location=map_location)
        if (
            isinstance(sd, dict)
            and "model" in sd
            and not any(k.startswith("aggregator.") for k in sd)
        ):
            sd = sd["model"]
        return self.model.load_state_dict(sd, strict=True)

    def apply_lora_adapters(self):
        if self.cfg.lora.enabled and not self._lora_applied:
            n = apply_lora(self.model.aggregator, self.cfg.lora)
            self._lora_applied = True
            return n
        return 0

    def freeze_for_finetune(self) -> dict:
        """Freeze everything except: LoRA adapters (the base projections stay
        frozen -- wrapping != unfreezing) and the DepthConditioner (which owns
        all new output-head channels / zero-init convs / gate)."""
        for name, p in self.model.named_parameters():
            p.requires_grad = ("lora_A" in name) or ("lora_B" in name)
        if self.conditioner is not None:
            for p in self.conditioner.parameters():
                p.requires_grad = True
        stats = param_stats(self)
        stats["base_attention_frozen"] = self.check_base_attention_frozen()
        return stats

    def check_base_attention_frozen(self) -> bool:
        """True iff every base attention projection matrix has requires_grad=False."""
        for name, p in self.model.aggregator.named_parameters():
            if ("attn" in name) and ("lora_A" not in name) and ("lora_B" not in name):
                if p.requires_grad:
                    return False
        return True

    # ------------------------------------------------------------------
    # depth gathering
    # ------------------------------------------------------------------
    def _gather_sparse_depth(self, views, images):
        """Stack per-view sparse depth + validity into [B,S,H,W]. Views without
        sparse depth contribute all-invalid (zero depth, zero mask) frames --
        'no measurement' is representable by construction."""
        B, S, _, H, W = images.shape
        depths, masks = [], []
        for view in views:
            d = view.get("sparse_depth")
            m = view.get("sparse_depth_mask")
            if d is None:
                d = images.new_zeros(B, H, W)
                m = images.new_zeros(B, H, W)
            elif m is None:
                m = (d > 0).to(d.dtype)
            depths.append(d.to(images.device))
            masks.append(m.to(device=images.device, dtype=images.dtype))
        depth = torch.stack(depths, dim=1)
        mask = torch.stack(masks, dim=1)
        return depth, mask

    def _conditioner_outputs(self, views, images):
        """Returns (depth_token_feats, depth_head_residuals) for this batch."""
        if self.conditioner is None:
            return None, None
        depth, mask = self._gather_sparse_depth(views, images)
        H, W = images.shape[-2:]
        if self.cfg.depth_cond.injection == "token":
            return self.conditioner(depth, mask), None
        sizes = dpt_fusion_sizes(H, W, self.patch_size)
        return None, self.conditioner(depth, mask, out_hw_list=sizes)

    # ------------------------------------------------------------------
    # encoder feature cache
    # ------------------------------------------------------------------
    def _cached_patch_tokens(self, views, images) -> Optional[torch.Tensor]:
        """Load (or compute-and-store) frozen patch-embed features.

        Frames are keyed by view["cache_key"] (a str for B==1, else a list of B
        strings). Keys must uniquely identify the *processed* RGB frame
        (sequence, frame index, resolution/crop); caching with augmentations
        that change pixels per epoch would poison the cache. If any view lacks
        a key, the whole batch falls back to the live encoder.
        """
        if self.cache is None:
            return None
        B, S, _, H, W = images.shape
        keys = []  # [S][B]
        for view in views:
            k = view.get("cache_key")
            if k is None:
                return None
            if isinstance(k, str):
                k = [k]
            if len(k) != B:
                return None
            keys.append(list(k))

        param = next(self.model.aggregator.patch_embed.parameters())
        loaded = {}
        missing = []
        for s in range(S):
            for b in range(B):
                t = self.cache.load(keys[s][b], device=images.device)
                if t is None:
                    missing.append((s, b))
                else:
                    loaded[(s, b)] = t.to(dtype=param.dtype)

        if missing:
            with torch.no_grad():
                imgs = torch.stack(
                    [images[b, s] for (s, b) in missing], dim=0
                )  # [M,3,H,W]
                feats = self.model.aggregator.embed_patches(
                    imgs.unsqueeze(1)
                )  # [M,P,C]
            for i, (s, b) in enumerate(missing):
                self.cache.save(keys[s][b], feats[i])
                loaded[(s, b)] = feats[i].to(dtype=param.dtype)

        # assemble in aggregator layout: [B*S, P, C] with frame-major flattening
        # matching images.reshape(B*S, ...) (b major, s minor)
        rows = [loaded[(s, b)] for b in range(B) for s in range(S)]
        return torch.stack(rows, dim=0)

    # ------------------------------------------------------------------
    # forward / inference
    # ------------------------------------------------------------------
    def forward(self, views, query_points: torch.Tensor = None):
        images = torch.stack([view["img"] for view in views], dim=0).permute(
            1, 0, 2, 3, 4
        )
        if images.dim() == 4:
            images = images.unsqueeze(0)
        token_feats, head_residuals = self._conditioner_outputs(views, images)
        patch_tokens = self._cached_patch_tokens(views, images)
        return self.model(
            views,
            query_points,
            patch_tokens=patch_tokens,
            depth_token_feats=token_feats,
            depth_head_residuals=head_residuals,
        )

    def inference(self, frames, query_points: torch.Tensor = None):
        """Streaming inference: per-frame conditioning (S=1 is the degenerate
        case of the same [B,S,H,W] contract; token feats enter before the KV
        cache each step)."""
        token_list, residual_list = None, None
        if self.conditioner is not None:
            token_list, residual_list = [], []
            for frame in frames:
                img = frame["img"]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images = img.unsqueeze(1)  # [B,1,3,H,W]
                token_feats, head_residuals = self._conditioner_outputs([frame], images)
                token_list.append(token_feats)
                residual_list.append(head_residuals)
        return self.model.inference(
            frames,
            query_points,
            depth_token_feats_list=token_list,
            depth_head_residuals_list=residual_list,
        )
