#!/bin/bash
#SBATCH --job-name=hypersim_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hypersim_download_%j.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Selective Hypersim download (~150GB measured, vs 1.57TB full dataset):
# only the render passes preprocess_hypersim.py consumes. The vendored
# downloader extracts matching members from Apple's per-scene zips via HTTP
# range requests -- nothing else is transferred. Resumable: existing files
# are skipped, so re-submitting continues.
#
# NOTE: the downloader ANDs all --contains words together, so each pass
# pattern needs its own invocation.

set -euo pipefail
mkdir -p logs

PY=/users/jdosch/miniconda3/envs/StreamVGGT/bin/python
REPO=/oscar/home/jdosch/MeTRIC
OUT=/gpfs/data/jtompki1/cli277/metric/hypersim

mkdir -p "$OUT"
cd "$REPO/datasets_download"

# global camera-parameters CSV (preprocess reads it from the dataset root;
# it ships in the ml-hypersim repo, not in the scene zips)
[ -f "$OUT/metadata_camera_parameters.csv" ] || curl -sS --fail \
    "https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv" \
    -o "$OUT/metadata_camera_parameters.csv"

for pattern in .color.hdf5 .depth_meters.hdf5 .render_entity_id.hdf5 _detail; do
    echo "=== pass: $pattern ==="
    $PY download_hypersim_subset.py -d "$OUT" -c "$pattern" --silent
done
echo "done: $(date)"
