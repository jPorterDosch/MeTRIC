#!/bin/bash
#SBATCH --job-name=hm_tokinj_headonly
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out

# =============================================================================
# ABLATION LADDER 4/4 -- NEGATIVE CONTROL: TOKEN injection + depth head only.
#
#   injection = TOKEN (sparse depth injected PRE-KV-cache, into the latent
#                      stream -- but the decoder is FROZEN)
#   trainable = depth head + DepthConditioner only (NO LoRA)
#
# Expected to underperform / collapse: token injection feeds depth into the
# decoder BEFORE attention, but with no LoRA the frozen decoder was never
# trained to read those tokens, so the head can't recover the signal
# downstream. This run exists to CONFIRM that failure -- i.e. that the token
# arm's gains (if any) in train_hammer_tokeninject_lora.sh come from decoder
# plasticity, not from the injection alone. Cheap (frozen backbone, no LoRA
# backward), so it's worth having as the fourth corner of the 2x2.
#
# If this run matches or beats the LoRA token arm, the LoRA is doing nothing
# useful for token injection -- an equally important (and surprising) result.
#
# Run:
#   sbatch experiments/hammer_finetune/train_hammer_tokeninject_headonly.sh
#   bash   experiments/hammer_finetune/train_hammer_tokeninject_headonly.sh  # allocated GPU
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
    `# --- conditioning arm: TOKEN injection, depth head only, NO LoRA -----` \
    --depth-cond.injection TOKEN \
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
