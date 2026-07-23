#!/bin/bash
#SBATCH --job-name=hm_tokinj_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out

# =============================================================================
# ABLATION LADDER 3/4 -- HYPOTHESIS: TOKEN injection + LoRA on decoder attention.
#
#   injection = TOKEN (sparse depth encoded to tokens injected PRE-KV-cache,
#                      into the decoder's latent space -- the proposed arm)
#   trainable = depth head + DepthConditioner + LoRA adapters on the decoder
#               attention projections (Q/K/V/O, rank 16, alpha 32; base frozen)
#
# The proposed method. Token injection feeds depth into the latent stream
# BEFORE attention, so it can only be used if the decoder is allowed to adapt
# -- hence LoRA is required here (a frozen decoder was never trained to read
# injected depth tokens; cf. the token+head-only negative control).
#
# Contrasts:
#   * vs train_hammer_headinject_lora.sh      -> PURE injection-point effect
#     (trainable set identical, only HEAD->TOKEN changes). THE key comparison.
#   * vs train_hammer_headinject_headonly.sh  -> full proposed-vs-baseline
#     (both axes), decomposed by the two runs above.
#
# ~2.5-3x baseline per-step cost; one SBATCH, under the 48h cap. Loss / epochs
# / seed identical to every ladder arm.
#
# Run:
#   sbatch experiments/hammer_finetune/train_hammer_tokeninject_lora.sh
#   bash   experiments/hammer_finetune/train_hammer_tokeninject_lora.sh  # allocated GPU
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer

# --- environment: the StreamVGGT conda env has torch/accelerate/tyro etc. ---
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
# expandable_segments: avoids fragmentation-class OOMs on L40S, where a small
# alloc can fail with memory free but non-contiguous. A frozen backbone is NOT
# immunity -- the token head-only arm OOM'd exactly this way -- so every arm
# sets it for parity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .secrets/wandb-personal.env

mkdir -p "$REPO/logs"

# finetune_depth.py resolves relative paths (and save_current_code's ".")
# against the CWD, so run from src/ like the other entrypoints
cd "$REPO/src"

python finetune_depth.py \
    --exp-group hammer_sweep \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: TOKEN injection, LoRA on decoder + depth head -` \
    `# TOKEN feeds depth into the latent stream pre-KV-cache; LoRA lets the`  \
    `# decoder actually consume it (defaults: targets Q/K/V/O, rank 16,`      \
    `# alpha 32). Trainable = LoRA adapters + depth head + conditioner.`      \
    --depth-cond.injection TOKEN \
    --depth-cond.heads DEPTH \
    --lora.enabled \
    --lora.rank 16 \
    --train.train-heads DEPTH \
    \
    `# --- loss: log-depth accuracy term + low confidence-reg weight --------` \
    --loss.depth-log-space \
    --loss.depth-alpha 0.02 \
    \
    `# --- train data: HAMMER only -----------------------------------------` \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.stride-range 1 20 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: HAMMER test split -------------------------------------` \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- optimization ------------------------------------------------------` \
    --batch-size 1 \
    --accum-iter 1 \
    --epochs 5 \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --warmup-epochs 0.5 \
    --weight-decay 0.05 \
    --amp 1 \
    --seed 42 \
    \
    `# --- cadence -----------------------------------------------------------` \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 8 \
    --print-freq 10
