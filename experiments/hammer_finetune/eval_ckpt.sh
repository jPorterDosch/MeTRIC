#!/bin/bash
#SBATCH --job-name=hm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=8:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x/%j.out

# =============================================================================
# ONE JOB, ALL TABLE ROWS: evaluates base StreamVGGT + both LoRA arms
# back-to-back under the sequential eval protocol (TEST split samples
# consecutive frames by construction, so tae/tae_sq are genuinely temporal).
#
# Each eval is finetune_depth.py with --epochs 0: no training, one val pass
# (val/hammer/*) + streaming eval (final_stream/hammer/*), no checkpoints
# written. Base runs without --resume (conditioning is zero-init no-op ==
# pretrained); the arms resume their checkpoint-best.pth.
#
# exp-group is hammer_eval (NOT hammer_sweep) so these runs are cleanly
# separated from the old shuffled-protocol sweep numbers in wandb.
#
#   sbatch experiments/hammer_finetune/eval_ckpt.sh
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer
CKPTS=$REPO/checkpoints/hammer_sweep

export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .secrets/wandb-personal.env

mkdir -p "$REPO/logs/hm_eval"
cd "$REPO/src"

COMMON=(
    --exp-group hammer_eval
    --pretrained "$REPO/ckpt/checkpoints.pth"
    --save-dir "$REPO/checkpoints"
    --depth-cond.heads DEPTH
    --lora.enabled
    --lora.rank 16
    --train.train-heads DEPTH
    --loss.depth-log-space
    --loss.depth-alpha 0.02
    --train-dataset.root "$DATA"
    --train-dataset.dataset HAMMER
    --train-dataset.stride-range 1 20
    --train-dataset.epoch-size 4500
    --train-dataset.highres-root None
    --val-dataset.root "$DATA"
    --val-dataset.dataset HAMMER
    --val-dataset.stride-range 1 1
    --val-dataset.epoch-size 1000
    --val-dataset.highres-root None
    --epochs 0
    --batch-size 1
    --accum-iter 1
    --lr 1e-5
    --min-lr 1e-7
    --warmup-epochs 0.5
    --weight-decay 0.05
    --amp 1
    --seed 42
    --val-freq 1
    --save-freq 0
    --num-workers 8
    --print-freq 10
)

echo "================ [1/3] base StreamVGGT ================"
python finetune_depth.py "${COMMON[@]}" \
    --depth-cond.injection HEAD

echo "================ [2/3] head inject + LoRA ================"
python finetune_depth.py "${COMMON[@]}" \
    --depth-cond.injection HEAD \
    --resume "$CKPTS/18075bb53f48343c/checkpoint-best.pth"

echo "================ [3/3] token inject + LoRA ================"
python finetune_depth.py "${COMMON[@]}" \
    --depth-cond.injection TOKEN \
    --resume "$CKPTS/b536d87d26e297e1/checkpoint-best.pth"

echo "================ all evals done ================"
