# arkit-scenes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p ~/scratch/data/arkit_scenes
python $SCRIPT_DIR/download_arkit_scenes.py raw \
    --download_dir ~/scratch/data/arkit_scenes \
    --video_id_csv raw/raw_train_val_splits.csv \
    --raw_dataset_assets lowres_wide lowres_depth lowres_wide_intrinsics lowres_wide.traj
