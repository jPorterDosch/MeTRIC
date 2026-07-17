"""Verify the zip-layout ARKitScenes raw tree after a download run.

Two tiers:
  * default (cheap, seconds): every scene zip has a readable central
    directory and at least one member -- catches truncation and non-zip
    garbage without reading the data.
  * --crc (heavy: reads and checksums EVERY byte; hours over a full tree):
    zipfile.testzip() per archive. Normally unnecessary -- curl --fail plus
    the .tmp+rename contract prevents truncated finals, and preprocessing
    CRC-checks members as it reads them -- but available for paranoia or
    after suspected filesystem trouble.

Usage:
    python verify_arkit_zips.py /path/to/arkit_scenes [--crc]

Exit code 0 iff no problems found.
"""

import argparse
import glob
import os
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("download_dir", help="root containing raw/<split>/<video_id>/")
    parser.add_argument(
        "--crc", action="store_true", help="full CRC pass (reads every byte)"
    )
    args = parser.parse_args()

    pattern = os.path.join(args.download_dir, "raw", "*", "*", "*.zip")
    zips = sorted(glob.glob(pattern))
    if not zips:
        print(f"no zips found under {pattern}")
        sys.exit(1)

    bad = []
    for i, path in enumerate(zips, 1):
        try:
            with zipfile.ZipFile(path) as zf:
                if not zf.namelist():
                    bad.append((path, "empty archive"))
                elif args.crc:
                    corrupt = zf.testzip()
                    if corrupt is not None:
                        bad.append((path, f"CRC mismatch in member {corrupt!r}"))
        except zipfile.BadZipFile as e:
            bad.append((path, f"unreadable: {e}"))
        if i % 2000 == 0:
            print(f"checked {i}/{len(zips)}...", flush=True)

    print(f"zips checked: {len(zips)}   problems: {len(bad)}")
    for path, why in bad[:50]:
        print(f"  {path}: {why}")
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
