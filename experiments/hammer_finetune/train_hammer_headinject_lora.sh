#!/bin/bash
#SBATCH --job-name=hm_headinj_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out

# =============================================================================
# ABLATION LADDER 2/4 -- BRIDGE: HEAD injection + LoRA on decoder attention.
#
#   injection = HEAD  (sparse depth injected post-KV-cache, into the depth
#                      head's DPT fusion -- the control injection point)
#   trainable = depth head + DepthConditioner + LoRA adapters on the decoder
#               attention projections (Q/K/V/O, rank 16, alpha 32; base
#               weights stay frozen)
#
# The de-confounding cell. It shares the injection point with the baseline and
# the trainable set with the token arm, so:
#   * vs train_hammer_headinject_headonly.sh -> PURE LoRA effect
#     (injection fixed at HEAD, only decoder plasticity added).
#   * vs train_hammer_tokeninject_lora.sh    -> PURE injection-point effect
#     (trainable set fixed at LoRA+head, only HEAD->TOKEN changes).
# Without this run, baseline-vs-token would confound "LoRA" with "token".
#
# ~2.5-3x the baseline's per-step cost (LoRA needs a decoder backward pass +
# grad_checkpoint recompute); ~one SBATCH, under the 48h cap. Loss/epochs/seed
# identical to every ladder arm.
#
# Run:
#   sbatch experiments/hammer_finetune/train_hammer_headinject_lora.sh
#   bash   experiments/hammer_finetune/train_hammer_headinject_lora.sh  # allocated GPU
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer

# --- environment: the StreamVGGT conda env has torch/accelerate/tyro etc. ---
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH

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
    `# --- conditioning arm: HEAD injection, LoRA on decoder + depth head --` \
    `# --lora.enabled wraps the attention projections (defaults: targets`     \
    `# Q/K/V/O, rank 16, alpha 32); train-heads DEPTH also unfreezes the`     \
    `# depth head. Trainable = LoRA adapters + depth head + conditioner.`     \
    --depth-cond.injection HEAD \
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
    --train-dataset.max-interval 20 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: HAMMER test split -------------------------------------` \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.max-interval 20 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- optimization ------------------------------------------------------` \
    --batch-size 2 \
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
