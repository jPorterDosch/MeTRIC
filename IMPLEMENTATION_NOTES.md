# MeTRIC Depth-Conditioning — Implementation Notes

Branch: `feat/depth-conditioning`. Spec: depth-conditioning module for StreamVGGT with
`head` (control) vs `token` (proposed) injection behind one config flag.

## Stage 0 — Recon (real numbers, replacing every ⚠️EST in the spec)

Read from `src/streamvggt/` (the causal / KV-cache model tree; `src/vggt/` is the
non-causal teacher used only for distillation in `train.py`).

| Quantity | Value | Source |
|---|---|---|
| Aggregator block count | **24 frame blocks + 24 global blocks** (alternating, `aa_order=["frame","global"]`, `aa_block_size=1` → 24 layer-pairs; the KV cache is per **global** block: `past_key_values` has 24 slots) | `src/streamvggt/models/aggregator.py` (`depth=24`) |
| Hidden dim (aggregator token dim) | **1024** (`embed_dim=1024`) | `aggregator.py`, `ckpt/config.json` |
| Num attention heads | **16** (head_dim 64, qk_norm=True, RoPE freq 100, LayerScale init 0.01) | `aggregator.py`, `layers/attention.py` |
| Encoder (patch-embed) token dim | **1024** — DINOv2 ViT-L/14 (`dinov2_vitl14_reg`), patch_size **14** | `aggregator.py __build_patch_embed__` |
| Special tokens | 1 camera + 4 register → `patch_start_idx = 5`; patch tokens start at index 5 | `aggregator.py` |
| DPT head input dim | **2048** = concat(frame-block out, global-block out), each 1024 | `models/streamvggt.py` (`dim_in=2*embed_dim`) |
| DPT taps | `intermediate_layer_idx = [4, 11, 17, 23]` into the 24-entry `aggregated_tokens_list` | `heads/dpt_head.py` |
| DPT per-scale channels | `out_channels = [256, 512, 1024, 1024]` after `projects`; fused to `features=256` by `scratch.layer{1..4}_rn` (3×3, stride 1, pad 1, **bias=False**) — these are the "first DPT fusion convs" | `heads/dpt_head.py` |
| DPT fusion spatial sizes | with `ph=H/14, pw=W/14`: scale0 `(4ph,4pw)`, scale1 `(2ph,2pw)`, scale2 `(ph,pw)`, scale3 `(⌈ph/2⌉,⌈pw/2⌉)` (stride-2 conv, k3 p1) | `heads/dpt_head.py resize_layers` |
| Default training clip length | **`num_views = 10`** (`config/finetune.yaml`); DPT head chunks frames at `frames_chunk_size=8` internally | `config/finetune.yaml`, `dpt_head.py` |
| Total params | see Stage 3/7 section below (measured, not estimated) | measured |

Other Stage-0 findings that shaped the implementation:

- **Two model trees.** `src/vggt/` (plain VGGT, no causal mask, no KV cache) and
  `src/streamvggt/` (causal global attention + per-global-block KV cache). The
  hypothesis is about the KV cache, so everything here targets `src/streamvggt/`.
  `src/finetune.py` currently fine-tunes `VGGT`; a new entrypoint
  `src/finetune_depth.py` fine-tunes the depth-conditioned StreamVGGT.
- **Checkpoint**: `ckpt/checkpoints.pth` is a raw StreamVGGT `state_dict`
  (1797 keys, fp32, ~5 GB) and loads with `strict=True` into `StreamVGGT()`.
  (`config/finetune.yaml` points at `../ckpt/model.pt`, which does not exist —
  the new config points at `../ckpt/checkpoints.pth`.)
- **Fused QKV.** Attention uses a single `nn.Linear(dim, 3*dim)` for QKV
  (`layers/attention.py`), so per-target (q/k/v) LoRA is implemented as
  low-rank updates added to the corresponding *output slices* of the fused
  projection (see `lora.py`); `o` wraps `attn.proj` as a standard LoRA linear.
- **QKV → cache path.** In the cached (streaming) path, K/V appended to
  `past_key_values` are the raw per-block `k`,`v` *after* the qkv projection
  (before q/k norm + RoPE) — so LoRA on k/v directly shapes what is cached.
- **Trainer stack**: hydra + accelerate (`bf16`), `dust3r.inference.loss_of_one_batch`
  calls `model(views, query_points)` where `views` is a list (length S) of dicts
  with `"img" [B,3,H,W]` (range mapped to [0,1] in the train loop).
- **Gradient checkpointing**: `finetune.py` calls `model.gradient_checkpointing_enable()`
  but neither model defines it (flag was always False, path never exercised). The
  depth-cond wiring implements checkpointing inside `Aggregator` directly
  (`torch.utils.checkpoint`, `use_reentrant=False`) behind `train.grad_checkpoint`.
- **No local dataset** is present under the configured `../data/train/...` roots on
  this machine, so the Stage-7 overfit check uses a synthetic clip (RGB + dense GT
  depth + simulated sparse depth), which is what the spec's "synthetic tensors"
  language allows.

## Deviations from the spec (and why)

1. **"Widen the first DPT fusion conv" is implemented as a parallel zero-init conv.**
   `conv([x; d], [W_x; W_d]) ≡ conv(x, W_x) + conv(d, W_d)` by linearity, so a separate
   zero-init `Conv2d(2, 256)` per scale whose output is added to `scratch.layer{i}_rn(x)`
   is *numerically identical* to widening `in_channels` by 2 with zero-init slices —
   while keeping the pretrained conv untouched/frozen and the checkpoint loading strict.
2. **Token-arm gating**: a learnable scalar gate initialized to 0 (spec offered
   zero-init projection *or* scalar gate). Only ONE zero is used — gate=0 with a
   normally-initialized projection — because zeroing both kills the gradient of each
   (`d/dproj ∝ gate = 0` and `d/dgate ∝ proj(x) = 0`). The gate value is also a useful
   scalar to log ("how open is the depth pathway").
3. **Identity encoder + token arm**: "raw passthrough" is realized as
   `pixel_unshuffle(14)` on the 2-channel (disparity, validity) map → `[2·14², ph, pw]`
   → one 1×1 conv to token dim. This is lossless w.r.t. the prepared depth (every pixel
   of both channels reaches the projection), which is the closest faithful reading of
   "identity" when a projection to token dim is unavoidable.
4. **Temporal attention no-op semantics**: implemented as a pre-LN residual block with
   **zero-init output projection** and a causal mask over S. At init it is an exact
   no-op for every S (not just S=1), which subsumes the Stage-6 check. Post-training,
   S=1 becomes a learned per-frame transform; the strict "attends only to itself ⇒
   passthrough" reading is not achievable with a standard attention block (V/O
   projections are not identity), and the zero-init residual is the conservative
   resolution.
5. **`log_depth` ordering**: `log1p` is applied **after** the fixed scaling
   (`disp * norm_constant_m`), i.e. `x = log1p(disp / (1/norm_constant_m))`. Same
   fixed monotone transform for every frame/sample ⇒ absolute scale is preserved.
6. **Head-arm injection targets both DPT heads (`depth_head` and `point_head`) by
   default** (configurable via `depth_cond.heads`): the goal is metric depth *and*
   metric pointmaps, and the control arm should get depth wherever the head can use it.
7. `camera_head` / `track_head` are untouched by injection in both arms.

## Stage log (all checks green; scripts under `tests/`, run with the
`StreamVGGT` conda env python from the repo root)

| Stage | Check | Result |
|---|---|---|
| 1 | head-injected vs pretrained, depth zeroed, tol 1e-4 | **PASS, max\|diff\| = 0.0 exactly** (also 0.0 with real sparse depth — zero-init convs are a strict no-op at init). Checkpoint loads `strict=True`. |
| 2 | masked pooling: empty cell → (0, frac 0); valid cell → true mean (not diluted); sizes match fusion scales | **PASS**; discovered scale sizes for 154×140 input: (44,40), (22,20), (11,10), (6,5) — verified against the real `DPTHead.projects+resize_layers`, incl. the ceil-div stride-2 scale |
| 3A | token arm + LoRA vs pretrained, depth zeroed, tol 1e-4 | **PASS, max\|diff\| = 0.0 exactly** (gate=0 and LoRA B=0 are both exact no-ops) |
| 3B | base attention frozen; trainable counts | **PASS** — see param table below |
| 3C | confound assertion fires on any diff besides `injection` | **PASS** (fires on `lora.rank` mismatch; head/token-only diff hashes are equal) |
| 4 | live vs cached encoder features, tol 1e-5 | **PASS, 0.0 exactly** (fp32 features stored verbatim; cold and warm paths both checked) |
| 5 | conv encoder end-to-end shapes both arms; zero-depth ≈ baseline | **PASS, 0.0 exactly for both arms** — head-arm preservation is exact (not approximate) because the per-scale projections *after* the conv stem are zero-init, so the stem's biases never reach the head at init |
| 6 | temporal S=1 no-op, tol 1e-5 | **PASS, 0.0 exactly** at init for S=1 *and* S=3 (zero-init residual out-proj); causality verified: with random temporal weights, frame 0 of an S=3 clip equals the S=1 run bit-exactly |
| 7 | 5-step overfit via `src/finetune_depth.py`, both arms, only the flag changed | **PASS** — head: 15.9776 → 10.2206, token: 15.9776 → 10.5469, both strictly monotone, grad checkpointing ON, real criterion (`ConfLoss(Regr3DPose)+FinetuneLoss`) on a synthetic geometrically-consistent clip. Step-0 losses are bit-identical across arms (both start exactly at the pretrained model). Startup confound check fires for a `lora.rank=8` run vs the head manifest. |

Extra (beyond required checks): streaming `inference()` with the KV cache +
token injection runs end-to-end (3 frames, per-frame conditioning enters
before the cache).

## Final parameter counts (measured, Stage 3/7)

| | total | trainable | % | LoRA | conditioner |
|---|---|---|---|---|---|
| token arm (identity enc, LoRA q/k/v/o r16 on all 48 blocks) | 1,263,231,405 | 6,693,889 | 0.530% | 6,291,456 | 402,433 |
| head arm (identity enc, same LoRA block) | 1,262,865,836 | 6,328,320 | 0.501% | 6,291,456 | 36,864 |

(The spec's "~950M → few M" estimate: real base is ~1.26B; trainable is ~6.3–6.7M.)
`base_attention_frozen` is asserted programmatically at every model build.

## Operational notes

- **Entrypoint**: `src/finetune_depth.py` (hydra config `config/finetune_depth.yaml`),
  run from `src/` like the existing trainers. `smoke_test=true` skips datasets and
  runs the 5-step synthetic overfit. Every run writes `manifest.json` (with the
  injection-exempt comparable hash) to its output dir; set
  `compare_with_manifest=<other run's manifest.json>` to hard-enforce the confound
  rule at startup.
- **Sparse depth inputs**: views may carry `sparse_depth` + `sparse_depth_mask`
  `[B,H,W]`; if absent during training, `simulate_sparse_depth` subsamples the GT
  `depthmap` (`depth_cond.sim_num_points` per frame). Views with no depth at all
  contribute all-invalid frames (zero signal + zero mask), which is representable
  by construction and distinct from "reading of 0 m".
- **Encoder cache keys**: `view["cache_key"]` must uniquely identify the *processed*
  RGB frame (sequence, frame idx, resolution/crop). Per-epoch pixel-changing
  augmentation (e.g. SeqColorJitter) would poison the cache — only enable
  `encoder_cache` with deterministic preprocessing. Batches lacking keys fall back
  to the live encoder automatically (paths are numerically identical, Stage 4).
- **Temporal + identity encoder** runs the S-attention on the raw 2-channel
  full-res map (dim-2 attention). It is functional and cheap but of limited use;
  the intended pairing is `temporal: attention` with `encoder: conv` (attention on
  the C-dim patch-grid latent). As spec'd, this axis exists to be ablated against
  `temporal: none`, not assumed beneficial.
- **Known repo quirk (pre-existing)**: with hydra 1.3 in 1.1-compat mode the process
  chdirs into `hydra.run.dir`; relative paths in the config resolve from there
  (same behavior as the existing `finetune.py`). Use absolute overrides when in doubt
  (the Stage-7 test does).
