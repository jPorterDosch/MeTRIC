#!/bin/bash
#SBATCH --job-name=hypersim_preprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hypersim_preprocess_%j.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Hypersim raw hdf5 passes -> processed {rgb.png, depth.npy, cam.npz} per
# frame (~230k files / ~300GB; inode-safe without zip packing). The
# preprocess loops scenes sequentially and skips nothing on rerun -- it
# overwrites cleanly, so resubmission after a failure is safe.

set -euo pipefail
mkdir -p logs

PY=/users/jdosch/miniconda3/envs/StreamVGGT/bin/python
REPO=/oscar/home/jdosch/MeTRIC
RAW=/gpfs/data/jtompki1/cli277/metric/hypersim
OUT=/gpfs/data/jtompki1/cli277/metric/processed_hypersim

export TQDM_MININTERVAL=60
cd "$REPO"

$PY datasets_preprocess/preprocess_hypersim.py \
    --hypersim_dir "$RAW" \
    --output_dir "$OUT"
echo "done: $(date)"
