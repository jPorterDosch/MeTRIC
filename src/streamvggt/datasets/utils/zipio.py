"""Zip-backed IO for the streamvggt datasets.

Scenes can be stored either as plain directories or as a single UNCOMPRESSED
(ZIP_STORED) archive per scene -- one inode instead of one per frame, with
random-access reads served straight from member byte ranges (no
decompression). read_bytes() accepts paths in either layout:

    <root>/<split>/<scene>/vga_wide/xxx.jpg        (plain file)
    <root>/<split>/<scene>.zip/vga_wide/xxx.jpg    (member of scene.zip)

The first ".zip/" component splits archive from member. Open handles are
cached per (pid, path): a ZipFile opened before a DataLoader fork must never
be used by both parent and child (the underlying fd's seek offset is shared
across processes), so each process lazily opens its own.
"""

import functools
import os
import zipfile

_ZIP_MARKER = ".zip/"


@functools.lru_cache(maxsize=64)
def _open_zip(pid, zip_path):
    # pid is part of the cache key ONLY so each forked process opens its own
    # handle; it is deliberately unused in the body.
    return zipfile.ZipFile(zip_path, "r")


def split_zip_path(path):
    """Split '<archive>.zip/<member>' into (archive, member); return
    (None, path) when the path has no .zip/ component (plain-file layout)."""
    path = os.fspath(path)
    idx = path.find(_ZIP_MARKER)
    if idx == -1:
        return None, path
    end = idx + len(".zip")
    return path[:end], path[end + 1 :]


def read_bytes(path):
    """Read a file either from disk or from inside a stored scene zip."""
    archive, member = split_zip_path(path)
    if archive is None:
        with open(path, "rb") as f:
            return f.read()
    try:
        return _open_zip(os.getpid(), archive).read(member)
    except KeyError:
        raise FileNotFoundError(
            f"member {member!r} not found in archive {archive!r}"
        ) from None


def listdir(path):
    """List entry names directly under `path`, for both layouts.

    Plain directory -> os.listdir. Virtual '<archive>.zip/<prefix>' -> the
    unique first path components of members under that prefix (files and
    subdirectories alike, mirroring os.listdir semantics)."""
    archive, prefix = split_zip_path(path)
    if archive is None:
        return os.listdir(path)
    zf = _open_zip(os.getpid(), archive)
    prefix = prefix.rstrip("/")
    prefix = prefix + "/" if prefix else ""
    names = set()
    for member in zf.namelist():
        if member.startswith(prefix) and member != prefix:
            names.add(member[len(prefix) :].split("/", 1)[0])
    return sorted(names)


def exists(path):
    """os.path.exists for both layouts: a virtual path exists when the
    archive exists and holds the member (or any member under it, so
    directory-style prefixes count too)."""
    archive, member = split_zip_path(path)
    if archive is None:
        return os.path.exists(path)
    if not os.path.isfile(archive):
        return False
    zf = _open_zip(os.getpid(), archive)
    member = member.rstrip("/")
    dir_prefix = member + "/"
    for name in zf.namelist():
        if name == member or name.startswith(dir_prefix):
            return True
    return False


def frames_root(scene_dir):
    """Resolve where a PROCESSED scene's frame files live: the scene's
    frames.zip when present (inode-safe layout, frames addressed as
    '<scene>/frames.zip/<stream>/<name>'), else the scene directory itself
    (extracted layout). Metadata npz files always live as real files in the
    scene directory, in both layouts."""
    zip_path = os.path.join(scene_dir, "frames.zip")
    if os.path.isfile(zip_path):
        return zip_path
    return scene_dir


def asset_root(scene_dir, asset):
    """Resolve a raw ARKitScenes per-asset location for both layouts.

    Extracted layout:  <scene_dir>/<asset>/            (a real directory)
    Zip layout:        <scene_dir>/<asset>.zip/<asset>  (Apple's zips prefix
                       their members with the asset name)

    Returns the path under which frame files live, or None if the asset is
    absent in both layouts."""
    zip_path = os.path.join(scene_dir, asset + ".zip")
    if os.path.isfile(zip_path):
        return f"{zip_path}/{asset}"
    plain = os.path.join(scene_dir, asset)
    if os.path.isdir(plain):
        return plain
    return None


class SceneZipWriter:
    """Write one scene's files into a single UNCOMPRESSED zip, atomically.

    Writes go to '<final>.tmp' and the archive is renamed to its final name
    only on clean exit, so a final-named zip is always complete (the same
    crash contract as the downloaders' .tmp+rename). Members are STORED so
    readers seek straight to the bytes; encoding (cv2.imencode etc.) is the
    caller's job. One writer per scene -- concurrent writes to a single
    archive are not supported and never needed (parallelism is across
    scenes).
    """

    def __init__(self, final_path):
        self.final_path = os.fspath(final_path)
        self.tmp_path = self.final_path + ".tmp"
        self._zf = zipfile.ZipFile(
            self.tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        )

    def writestr(self, member_name, data):
        self._zf.writestr(member_name, data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._zf.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
        else:
            os.remove(self.tmp_path)
        return False
