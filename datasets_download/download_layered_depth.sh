# layered depth
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p ~/scratch/data/layered_depth_syn

python $SCRIPT_DIR/download_layered_depth.py