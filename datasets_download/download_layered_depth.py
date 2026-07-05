#!/usr/bin/env python3
"""
Download princeton-vl/LayeredDepth-Syn from the Hugging Face Hub.

The dataset is stored as parquet shards (~24.7 GB total: 14.8k train rows +
500 validation rows). Each row contains one RGB image plus up to 8 layered
depth maps (depth_1.png ... depth_8.png) and a __key__ id.

Usage:

    # everything
    python download_layereddepth_syn.py --out ~scratch/data/LayeredDepth-Syn

    # just the train split (parquet files live under a train/ prefix)
    python download_layereddepth_syn.py --out ~/scratch/data/LayeredDepth-Syn --split train

Notes:
  * The dataset is public (BSD-3-Clause) so no token is needed. If you hit
    rate limits, pass --token or set HF_TOKEN in your environment.
  * snapshot_download resumes automatically -- re-run it if it gets cut off.
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Download LayeredDepth-Syn.")
    p.add_argument("--out", default=os.path.expanduser("~/scratch/data/layered_depth_syn"),
                   help="Local directory to download into.")
    p.add_argument("--split", choices=["train", "validation"], default=None,
                   help="Only download one split. Omit to get everything.")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (optional; only for rate limits).")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download workers.")
    p.add_argument("--no-fast", action="store_true",
                   help="Disable hf_transfer accelerated downloads.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Turn on the legacy hf_transfer Rust downloader ONLY if that package is
    # actually installed. In huggingface_hub 1.x the Xet backend (hf-xet) is a
    # core dependency and accelerates downloads by default, so this is optional.
    # Setting the env var without hf_transfer installed makes hf_hub error out,
    # hence the guard. Must happen before huggingface_hub is imported.
    if not args.no_fast:
        import importlib.util
        if importlib.util.find_spec("hf_transfer") is not None:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    # allow_patterns lets us pull only one split's parquet files if requested.
    allow_patterns = None
    if args.split:
        allow_patterns = [f"{args.split}/*", f"data/{args.split}*", f"*{args.split}*.parquet"]

    path = snapshot_download(
        repo_id="princeton-vl/LayeredDepth-Syn",
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=allow_patterns,
        max_workers=args.workers,
        token=args.token,
    )
    print(f"\nDone. Files are in: {path}")


if __name__ == "__main__":
    main()