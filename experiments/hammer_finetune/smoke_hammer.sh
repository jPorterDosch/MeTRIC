#!/bin/bash
#SBATCH --job-name=hammer_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --constraint="a5000|l40s"
#SBATCH --time=1:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune_smoke/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune_smoke/%j.out

# =============================================================================
# smoke_hammer.sh -- 10-20 minute end-to-end sanity run of the HAMMER
# head-injection pipeline (same arm/flags as train_hammer.sh, tiny sizes:
# 10 train clips/epoch, 5 val clips, 5 epochs, lr 1e-4). Confirms with the
# CURRENT code that: data loads -> train forward/backward runs -> loss is
# finite and trends down -> per-epoch val (AbsRel/delta1/TAE) runs -> the
# final streaming (KV-cache) eval runs -> checkpoints + manifest + wandb all
# write. Checkpoints go to scratch; delete the dir after reading the result:
#   rm -rf /oscar/scratch/jdosch/checkpoints_smoke
#
# Execute:  sbatch /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/smoke_hammer.sh
#
# PASS = job completes; log shows loss decreasing over the 5 epochs, one
# "Val Epoch: [N]" block per epoch, a final "Streaming eval:" block, and
# checkpoint-last/best/final in the scratch dir. NOTE: the train sampler
# draws a fresh 10-clip subset each epoch (epoch-seeded), so expect a noisy
# downward loss trend, not textbook single-batch overfitting.
#
# The constraint accepts a5000 (idle nodes available, 24 GB Ampere, native
# bf16 -- the batch-1 config fit in 24 GB on the Quadro) OR l40s, so it
# starts without waiting on the busy L40S pool.
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer

export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH

# absolute path so sbatch works from any directory; deliberately NOT setting
# WANDB_RUN_ID/WANDB_RESUME here -- this must never append to the real run
source "$REPO/.secrets/wandb-personal.env"

mkdir -p "$REPO/logs/hammer_finetune_smoke"

cd "$REPO/src"

python finetune_depth.py \
    --exp-name hammer_smoke \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir /oscar/scratch/jdosch/checkpoints_smoke \
    \
    `# same head-injection arm as the real run`                               \
    --depth-cond.injection HEAD \
    --depth-cond.heads DEPTH \
    --lora.no-enabled \
    --train.train-heads DEPTH \
    \
    `# tiny data: 10 train clips per epoch, 5 deterministic val clips`        \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.max-interval 20 \
    --train-dataset.epoch-size 10 \
    --train-dataset.highres-root None \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.max-interval 20 \
    --val-dataset.epoch-size 5 \
    --val-dataset.highres-root None \
    \
    `# lr 1e-4 (10x the real run) so 50 total steps show a visible drop;`     \
    `# save-freq 0 disables mid-epoch saves (at 10 steps/epoch the 10%`       \
    `# cadence would otherwise write 5 GB every single step)`                 \
    --batch-size 1 \
    --epochs 5 \
    --lr 1e-4 \
    --save-freq 0 \
    --val-freq 1 \
    --num-workers 6 \
    --print-freq 1
