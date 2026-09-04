"""Rename cached masks after the photos they describe have moved.

The cache key used to be the absolute path of the photo, so moving a dataset - or cloning
the repo somewhere else - threw away every segmented mask although not one pixel had
changed. Keys are relative to the project now; this rewrites entries left under an old
identity so the GPU work survives the move.

    tools/migrate_mask_cache.py ~/3DPrint/tools/data/printcad/PrintCAD

The argument is where the photos used to live. Size and mtime survive a move, so an old key
can be recomputed exactly and the file renamed in place.
"""
import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from photo2fcstd.settings import cache_dir, data_dir  # noqa: E402
from photo2fcstd.trace import MASK_VERSION, TRIM_APPENDAGE, cache_identity  # noqa: E402

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")


def key(identity, size, mtime, version, trim):
    raw = "%s|%d|%d" % (identity, size, int(mtime))
    if version is not None:
        raw += "|v%s|t%.4f" % (version, trim)
    return hashlib.sha1(raw.encode()).hexdigest()


def photos(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.isfile(p) and p.lower().endswith(PHOTO_SUFFIXES)]


def migrate(old_root, new_root, versions, trims, dry_run):
    masks = cache_dir("masks")
    found = photos(new_root)
    moves = [
        (os.path.join(masks, key(os.path.join(old_root, os.path.relpath(p, new_root)),
                                 os.stat(p).st_size, os.stat(p).st_mtime, v, t) + ".npz"),
         os.path.join(masks, key(cache_identity(p), os.stat(p).st_size, os.stat(p).st_mtime, v, t) + ".npz"))
        for p in found for v in versions for t in trims
    ]
    pending = [(old, new) for old, new in moves if old != new
               and os.path.exists(old) and not os.path.exists(new)]
    if not dry_run:
        [os.rename(old, new) for old, new in pending]
    return len(found), len(pending)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_root", help="where the photos used to live")
    parser.add_argument("--new-root", default=None, help="where they live now (default: the configured dataset)")
    parser.add_argument("--versions", default="none,1,2,%s" % MASK_VERSION,
                        help="mask versions to migrate, comma separated; 'none' for unversioned keys")
    parser.add_argument("--trims", default="0.0,%s" % TRIM_APPENDAGE,
                        help="appendage-trim settings to migrate, comma separated")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    new_root = os.path.realpath(args.new_root or data_dir())
    old_root = os.path.realpath(os.path.expanduser(args.old_root))
    if not os.path.isdir(new_root):
        parser.error("no photos at %s" % new_root)

    versions = [None if v == "none" else v for v in args.versions.split(",") if v]
    trims = sorted({float(t) for t in args.trims.split(",") if t})
    scanned, moved = migrate(old_root, new_root, versions, trims, args.dry_run)
    print("%d photos under %s; %d cache entries %s"
          % (scanned, new_root, moved, "would move" if args.dry_run else "moved"))


if __name__ == "__main__":
    run()
