#!/usr/bin/env python3
"""
Download the HAMMER RGB-D dataset (polarization-camera subset).

The official archive is a single ~170 GB zip:

    https://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip

of which the CUT3R/DUSt3R integration only needs the polarization (RGB)
camera's rgb/, _gt/ (ground-truth depth), _pose/ and intrinsics.txt per
sequence (~24 GB uncompressed). The TUM server supports HTTP range requests,
so this script reads the zip's central directory remotely, selects only the
needed members, and downloads them in coalesced byte ranges: ~24 GB
transferred instead of 170 GB, and the zip itself is never stored on disk.

Fail-fast guarantees:
  * the member list is validated against the known archive contents
    (64 sequences, rgb/_gt/_pose frame counts equal per sequence) BEFORE
    any bulk download starts;
  * every extracted file is CRC32-checked against the archive index and
    written atomically (tmp file + rename), so an interrupted run never
    leaves a corrupt file behind;
  * re-running resumes: members whose output file already exists with the
    exact expected size are skipped.

Usage:
    python download_hammer.py --out ~/scratch/data/hammer

The output layout is identical to extracting the zip (sceneX_trajY_Z/
polarization/...), so datasets_preprocess/preprocess_hammer.py works the same
on a full manual extraction (wget + unzip) if you ever need other modalities.
"""

import argparse
import os
import os.path as osp
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import namedtuple

from tqdm import tqdm

HAMMER_URL = (
    "https://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/"
    "_dataset_processed.zip"
)
# Known contents of the official archive; a mismatch means a changed or
# corrupt upstream file and aborts before anything is downloaded.
EXPECTED_NUM_SEQUENCES = 64
NEEDED_RE = re.compile(
    r"^scene[^/]+/polarization/(?:rgb|_gt|_pose)/[^/]+$"
    r"|^scene[^/]+/polarization/intrinsics\.txt$"
)
GAP_TOL = 1 << 20  # merge members into one request across gaps up to 1 MB
TAIL_SLACK = 70000  # local header (30) + max name/extra fields, rounded up
RETRIES = 5

Member = namedtuple("Member", "name method crc csize usize lho")


def fetch_range(url, start, end, expected_len=None):
    """GET bytes [start, end] (inclusive) with retries; fails on short reads."""
    if expected_len is None:
        expected_len = end - start + 1
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status != 206:
                    raise RuntimeError(
                        f"server did not honor range request (HTTP {resp.status})"
                    )
                data = resp.read()
            if len(data) != expected_len:
                raise RuntimeError(f"short read: {len(data)} of {expected_len} bytes")
            return data
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
            last_err = e
            wait = 2**attempt
            print(f"range {start}-{end} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"range {start}-{end} failed after {RETRIES} attempts: {last_err}")


def read_central_directory(url):
    """Locate and fetch the zip64 central directory via range requests."""
    # EOCD (22) + zip64 locator (20) + zip64 EOCD (56) live at the very end;
    # fetch a generous tail in case of a trailing archive comment.
    tail_len = 66 * 1024
    req = urllib.request.Request(url, headers={"Range": f"bytes=-{tail_len}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206:
            raise RuntimeError("server does not support HTTP range requests")
        tail = resp.read()
        content_range = resp.headers.get("Content-Range", "")
    total_size = int(content_range.rsplit("/", 1)[-1])

    z64 = tail.rfind(b"PK\x06\x06")
    if z64 == -1:
        raise RuntimeError("zip64 end-of-central-directory record not found")
    (_, _, _, _, _, _, _, n_total, cd_size, cd_offset) = struct.unpack(
        "<IQHHIIQQQQ", tail[z64 : z64 + 56]
    )
    print(
        f"archive: {total_size / 1e9:.1f} GB, {n_total} members, "
        f"central directory {cd_size / 1e6:.1f} MB"
    )
    cd = fetch_range(url, cd_offset, cd_offset + cd_size - 1)

    members = []
    i = 0
    while i < len(cd):
        if cd[i : i + 4] != b"PK\x01\x02":
            raise RuntimeError(f"corrupt central directory at byte {i}")
        method = struct.unpack("<H", cd[i + 10 : i + 12])[0]
        crc, csize, usize = struct.unpack("<III", cd[i + 16 : i + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[i + 28 : i + 34])
        lho = struct.unpack("<I", cd[i + 42 : i + 46])[0]
        name = cd[i + 46 : i + 46 + nlen].decode("utf-8")
        extra = cd[i + 46 + nlen : i + 46 + nlen + elen]
        j = 0
        while j + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[j : j + 4])
            if hid == 1:  # zip64: only the maxed-out fields are present, in order
                body = extra[j + 4 : j + 4 + hsz]
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[k : k + 8])[0]
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[k : k + 8])[0]
                    k += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", body[k : k + 8])[0]
                    k += 8
            j += 4 + hsz
        members.append(Member(name, method, crc, csize, usize, lho))
        i += 46 + nlen + elen + clen

    if len(members) != n_total:
        raise RuntimeError(
            f"parsed {len(members)} central-directory entries, expected {n_total}"
        )
    return members, cd_offset


def select_members(members):
    """Filter to the polarization subset and validate against known contents."""
    needed = [m for m in members if NEEDED_RE.match(m.name)]
    counts = {}  # seq -> {subdir: n}
    for m in needed:
        seq, _, rest = m.name.split("/", 2)
        sub = rest.split("/", 1)[0] if "/" in rest else rest
        counts.setdefault(seq, {}).setdefault(sub, 0)
        counts[seq][sub] += 1
    if len(counts) != EXPECTED_NUM_SEQUENCES:
        raise RuntimeError(
            f"expected {EXPECTED_NUM_SEQUENCES} sequences in the archive, "
            f"found {len(counts)}; changed upstream archive?"
        )
    for seq, c in sorted(counts.items()):
        if c.get("intrinsics.txt") != 1:
            raise RuntimeError(f"{seq}: intrinsics.txt missing in archive")
        if not (c.get("rgb", 0) == c.get("_gt", 0) == c.get("_pose", 0) > 0):
            raise RuntimeError(f"{seq}: misaligned frame counts in archive: {c}")
    return needed


def build_groups(members, archive_end, chunk_bytes):
    """Coalesce members (sorted by offset) into few large byte-range requests."""
    members = sorted(members, key=lambda m: m.lho)
    groups = []
    for m in members:
        end_est = m.lho + 30 + len(m.name.encode()) + m.csize
        if groups:
            last = groups[-1]
            gap = m.lho - last["end_est"]
            if gap <= GAP_TOL and end_est - last["start"] <= chunk_bytes:
                last["members"].append(m)
                last["end_est"] = max(last["end_est"], end_est)
                continue
        groups.append({"start": m.lho, "end_est": end_est, "members": [m]})
    for g in groups:
        g["fetch_end"] = min(g["end_est"] + TAIL_SLACK, archive_end) - 1
    return groups


def extract_member(buf, group_start, member, out_root):
    off = member.lho - group_start
    if buf[off : off + 4] != b"PK\x03\x04":
        raise RuntimeError(f"{member.name}: local file header not found")
    nlen, elen = struct.unpack("<HH", buf[off + 26 : off + 30])
    dstart = off + 30 + nlen + elen
    raw = buf[dstart : dstart + member.csize]
    if len(raw) != member.csize:
        raise RuntimeError(f"{member.name}: fetched range too short")
    if member.method == 8:
        content = zlib.decompress(raw, -15)
    elif member.method == 0:
        content = raw
    else:
        raise RuntimeError(f"{member.name}: unsupported compression {member.method}")
    if len(content) != member.usize:
        raise RuntimeError(
            f"{member.name}: decompressed {len(content)} bytes, "
            f"expected {member.usize}"
        )
    if zlib.crc32(content) != member.crc:
        raise RuntimeError(f"{member.name}: CRC mismatch")

    out_path = osp.join(out_root, member.name)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="Download HAMMER (RGB subset).")
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/scratch/data/hammer"),
        help="Output directory (zip-identical layout).",
    )
    parser.add_argument("--url", default=HAMMER_URL)
    parser.add_argument(
        "--chunk_mb",
        type=int,
        default=128,
        help="Maximum size of one coalesced range request.",
    )
    args = parser.parse_args()

    members, archive_end = read_central_directory(args.url)
    needed = select_members(members)
    total_bytes = sum(m.usize for m in needed)
    print(
        f"selected {len(needed)} members "
        f"({total_bytes / 1e9:.1f} GB uncompressed) out of {len(members)}"
    )

    os.makedirs(args.out, exist_ok=True)
    todo = [
        m
        for m in needed
        if not (
            osp.isfile(osp.join(args.out, m.name))
            and os.path.getsize(osp.join(args.out, m.name)) == m.usize
        )
    ]
    if len(todo) < len(needed):
        print(f"resume: {len(needed) - len(todo)} members already downloaded")

    groups = build_groups(todo, archive_end, args.chunk_mb << 20)
    fetch_bytes = sum(g["fetch_end"] - g["start"] + 1 for g in groups)
    with tqdm(
        total=fetch_bytes, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for g in groups:
            buf = fetch_range(args.url, g["start"], g["fetch_end"])
            for m in g["members"]:
                extract_member(buf, g["start"], m, args.out)
            pbar.update(len(buf))

    # final on-disk check, so a bad state can never look like success
    missing = [
        m.name
        for m in needed
        if not osp.isfile(osp.join(args.out, m.name))
        or os.path.getsize(osp.join(args.out, m.name)) != m.usize
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} members missing/wrong size after download, "
            f"e.g. {missing[:3]}"
        )
    n_seqs = len(
        [d for d in os.listdir(args.out) if d.startswith("scene")]
    )
    print(
        f"HAMMER download successful: {n_seqs} sequences, "
        f"{len(needed)} files, {total_bytes / 1e9:.1f} GB in {args.out}"
    )


if __name__ == "__main__":
    main()
