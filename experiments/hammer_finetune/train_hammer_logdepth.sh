#!/bin/bash
#SBATCH --job-name=hammer_logdepth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out

# =============================================================================
# train_hammer_logdepth.sh -- variant of train_hammer.sh with two changes:
#
#   1. --loss.depth-log-space : the depth accuracy term runs on log-depth
#      (|log pred - log gt| ~ relative error) instead of raw metres, so the far
#      background stops dominating the L1 while the metric scale penalty is
#      kept (this is NOT scale-invariant SILog). Near objects and far walls now
#      contribute by RELATIVE error, weighted equally.
#
#   2. --loss.depth-alpha 0.02 : low weight on the confidence regularizer
#      (-alpha*log sigma). The 0.1 default let the confidence inflate and drove
#      the training loss negative / the val loss up (see the 39c6... run) while
#      AbsRel stayed ~0.055; a smaller alpha keeps the confidence term from
#      swamping the accuracy signal. (conf ~ alpha/err, so lower alpha -> lower,
#      better-behaved confidences.)
#
# Everything else is identical to train_hammer.sh: HEAD injection, depth head +
# conditioner trainable, no LoRA. New --exp-name so the run gets its own config
# hash / directory (the loss knobs are part of the experiment identity, so the
# hash differs from the baseline run regardless).
#
# Run:
#   sbatch /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/train_hammer_logdepth.sh
#   bash   /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/train_hammer_logdepth.sh   # on an allocated GPU
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
    --exp-name hammer_depth_cond_head_logdepth \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: HEAD injection, depth head only (as baseline) --` \
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
    --epochs 10 \
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
