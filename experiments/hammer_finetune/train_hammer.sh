#!/bin/bash
#SBATCH --job-name=hammer_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out

# =============================================================================
# train_hammer.sh -- full fine-tuning run of the depth-conditioned StreamVGGT
# (MeTRIC) on the HAMMER dataset, via the existing src/finetune_depth.py
# entrypoint (tyro CLI). This run tests the HEAD-injection arm: sparse depth
# is injected directly into the depth head's DPT fusion, with NO LoRA
# adapters -- only the depth head (+ the DepthConditioner) trains.
#
# How to execute (from anywhere):
#   sbatch /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/train_hammer.sh
# or, on a node that already has a GPU allocated:
#   bash   /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/train_hammer.sh
#
# Data: /oscar/scratch/jdosch/data/processed_hammer (46 train / 18 test
# sequences, produced by datasets_preprocess/preprocess_hammer.py and verified
# by tests/hammer_dataset_smoke.py). The exact CLI below was validated end to
# end on 2026-07-13: tyro parses it, and both HAMMER datasets build
# (train: 11,179 view-groups; val: 2,774).
#
# Outputs: checkpoints + manifest.json under
#   ${REPO}/checkpoints/hammer_depth_cond_head_<experiment-id>/
# (finetune_depth names the run dir by a SHA over the config and fails fast if
# it already exists, so a finished experiment is never silently re-run).
# Metrics stream to wandb project "MeTRIC"; export WANDB_MODE=offline first if
# the compute node has no internet.
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
    --exp-name hammer_depth_cond_head \
    \
    `# --- model / checkpointing -------------------------------------------` \
    `# pretrained StreamVGGT weights; the whole backbone stays frozen`        \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: HEAD injection, depth head only ---------------` \
    `# inject the sparse depth directly into the DPT fusion of the depth`     \
    `# head (injection HEAD = the control arm; heads DEPTH restricts the`     \
    `# injection to the depth head, the point head gets none). No LoRA:`      \
    `# --lora.no-enabled skips wrapping the attention projections, so the`    \
    `# trainable params are exactly the depth head + the DepthConditioner`    \
    `# (train.train-heads DEPTH unfreezes only the depth head; the point`     \
    `# head stays frozen). Encoder/temporal/norm stay at their defaults`      \
    `# (identity encoder, no temporal mixing, fixed 10 m disparity norm).`    \
    --depth-cond.injection HEAD \
    --depth-cond.heads DEPTH \
    --lora.no-enabled \
    --train.train-heads DEPTH \
    \
    `# --- train data: HAMMER only -----------------------------------------` \
    `# dataset takes the DatasetName MEMBER name (HAMMER, not hammer);`       \
    `# max-interval 20 is the HAMMER loader's own default sampling stride;`   \
    `# epoch-size 4500 mirrors the per-dataset slice of the original recipe;` \
    `# highres-root None because that knob only applies to ARKitScenes`       \
    `# lowres (the tyro default carries ARKitScenes entries, so it must be`   \
    `# cleared explicitly). Shared knobs stay at their defaults: num_views`   \
    `# 10, the 10-way aspect-ratio resolution list, aug_crop 16, and the`     \
    `# SeqColorJitter transform -- same augmentation the ARKitScenes recipe`  \
    `# uses.`                                                                 \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.max-interval 20 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: HAMMER test split -------------------------------------` \
    `# val defaults: split test, num_views 4, single (518, 392) resolution,`  \
    `# seed 42 -> the same deterministic clip set every epoch; epoch-size`    \
    `# 1000 mirrors the original recipe's test slice`                         \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.max-interval 20 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- optimization ------------------------------------------------------` \
    `# batch-size 4: streaming_eval gets its own batch-1 loader now, so the`  \
    `# train/val batch is unconstrained. 4 clips x 10 views assumes a`        \
    `# >=40GB card -- plausible because the frozen backbone (no LoRA, head`   \
    `# arm) stores no activations for backward -- but it is NOT memory-`      \
    `# profiled: drop to 2 or 1 if the job OOMs. lr 1e-5 -> cosine to`        \
    `# min-lr 1e-7 with 0.5-epoch warmup, AdamW wd 0.05, bf16 autocast`       \
    `# (amp 1), grads clipped to 1.0 in the loop`                             \
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
    `# validate (loss + AbsRel/delta1/TAE) every epoch and keep the best`     \
    `# checkpoint; checkpoint-last every 10% of an epoch so a preemption`     \
    `# loses little; num-workers matches the cpus-per-task request above`     \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 8 \
    --print-freq 10
