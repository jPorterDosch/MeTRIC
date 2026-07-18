#!/bin/bash
#SBATCH --job-name=hm_headinj_headonly
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out

# =============================================================================
# ABLATION LADDER 1/4 -- BASELINE: HEAD injection + depth head only (NO LoRA).
#
#   injection = HEAD  (sparse depth injected post-KV-cache, into the depth
#                      head's DPT fusion -- the control arm)
#   trainable = depth head + DepthConditioner only (backbone frozen, no LoRA)
#
# This is the cheap reference the other arms are measured against:
#   * vs train_hammer_headinject_lora.sh -> isolates the LoRA effect
#     (injection point held fixed at HEAD).
#
# Loss is log-depth + low confidence-reg (--loss.depth-log-space,
# --loss.depth-alpha 0.02), identical across every ladder arm so only the
# injection/trainable axes vary. epochs=5: HAMMER's metric floor is ~epoch 4
# (~15-18k steps); 10 epochs was pure overfit tail. Selection is on
# absrel_metric, so the best checkpoint is picked regardless of the knee.
#
# Run:
#   sbatch experiments/hammer_finetune/train_hammer_headinject_headonly.sh
#   bash   experiments/hammer_finetune/train_hammer_headinject_headonly.sh  # allocated GPU
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
    `# --- conditioning arm: HEAD injection, depth head only, NO LoRA ------` \
    --depth-cond.injection HEAD \
    --depth-cond.heads DEPTH \
    --lora.no-enabled \
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
