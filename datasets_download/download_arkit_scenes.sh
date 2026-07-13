# arkit-scenes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOWNLOAD_DIR=/oscar/data/jtompki1/cli277/metric/arkit_scenes
mkdir -p "$DOWNLOAD_DIR"

# Reproduce the DUSt3R/StreamVGGT ARKitScenes recipe: two variants from ONE download of
# the full raw split. Both variants share the 640x480 vga_wide RGB + vga_wide_intrinsics
# + lowres_wide.traj and differ only in the depth source:
#   * lowres  -> preprocess_arkitscenes.py         (vga_wide + lowres_depth  LiDAR)      => processed_arkitscenes/
#   * highres -> preprocess_arkitscenes_highres.py (vga_wide + highres_depth laser GT)   => processed_arkitscenes_highres/
# highres_depth only exists for the ~2257 "upsampling" scenes; the downloader auto-skips
# it on the rest (download_arkit_scenes.py:66-70). The highres preprocess then "owns"
# those scenes and the lowres loader excludes them at load time (arkitscenes.py:85-90),
# so the two variants partition the scenes without overlap.
python $SCRIPT_DIR/download_arkit_scenes.py raw \
    --download_dir "$DOWNLOAD_DIR" \
    --video_id_csv raw/raw_train_val_splits.csv \
    --raw_dataset_assets highres_depth lowres_depth vga_wide vga_wide_intrinsics lowres_wide.traj \
    --num_workers 8
