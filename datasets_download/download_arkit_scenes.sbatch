#!/bin/bash
#SBATCH --job-name=arkit_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/arkit_download_%j.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Download the raw ARKitScenes assets (lowres+highres recipe) with parallel
# workers. Resumable: already-downloaded files/extracted dirs are skipped, so
# re-submitting after a timeout or failure continues where it left off.

set -euo pipefail
mkdir -p logs

REPO=/oscar/home/jdosch/MeTRIC
# the .sh calls bare `python`; put the StreamVGGT env (pandas) first on PATH
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH

cd "$REPO/datasets_download"
NUM_WORKERS=16 bash download_arkit_scenes.sh
