# HAMMER (https://github.com/Junggy/HAMMER-dataset)
# Downloads only the polarization (RGB) camera subset needed for CUT3R
# (~24 GB of the 170 GB official zip) via HTTP range requests.
# Re-run to resume an interrupted download.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p ~/scratch/data/hammer
python $SCRIPT_DIR/download_hammer.py --out ~/scratch/data/hammer
