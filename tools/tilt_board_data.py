"""Turn a board capture into tilt labels, which is the one thing renders could not provide.

A part lying flat in the cleared middle of the target has its face normal parallel to the board's,
so the solved board pose *is* the tilt label - exact, free, and on a real photograph rather than a
render. That is the whole reason to shoot with the board once: not to use the check, to train it.

The assumption is the part lies flat with the face up. A part stood on edge is mislabelled and
there is no way to detect that here, so shoot them lying down.
"""
import os, sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import capture, embed, tilt_model  # noqa: E402
from photo2fcstd.trace import load  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".heic")


def photos_in(paths):
    out = []
    for a in paths:
        if os.path.isdir(a):
            out += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.lower().endswith(EXT)]
        elif a.lower().endswith(EXT):
            out.append(a)
    return out


def label_of(path):
    """The board's normal in camera coordinates, and the tilt it implies."""
    solved = capture.pose(load(path))
    if solved is None:
        return None
    R, _ = cv2.Rodrigues(np.asarray(solved["rvec"], float))
    n = R[:, 2]
    n = n * np.sign(n[2] if n[2] != 0 else 1)
    return n, float(np.degrees(np.arccos(min(abs(n[2]), 1.0)))), solved.get("reprojection_px")


def main(*args):
    paths = photos_in(args or ["."])
    out = os.path.join(ROOT, "data", "tilt_board.npz")
    if not paths:
        raise SystemExit("no photographs found; give a directory of board captures")
    rows, names = [], []
    for path in paths:
        got = label_of(path)
        if got is None:
            print("  %-40s no board pose" % os.path.basename(path))
            continue
        normal, tilt, err = got
        crop = tilt_model.part_crop(path)
        if crop is None:
            print("  %-40s no part found" % os.path.basename(path))
            continue
        rows.append((crop, normal, tilt))
        names.append(os.path.basename(os.path.dirname(path)) or "set")
        print("  %-40s tilt %5.1f deg   reprojection %s"
              % (os.path.basename(path), tilt, "%.2f px" % err if err else "-"))
    if len(rows) < 20:
        print("\n  only %d labelled frames; thirty to fifty is the ask, and they must span "
              "0 to 30 degrees" % len(rows))
    if not rows:
        raise SystemExit("nothing labelled")
    crops = [r[0] for r in rows]
    X = np.concatenate([embed.embed_images(crops[i:i + 16]) for i in range(0, len(crops), 16)])
    tilts = np.array([r[2] for r in rows])
    np.savez(out, obj=X, normal=np.array([r[1] for r in rows]),
             part=np.array([abs(hash(n)) % 10 ** 8 for n in names]), name=np.array(names))
    print("\n  %d frames, tilt %.0f to %.0f deg over %d groups"
          % (len(rows), tilts.min(), tilts.max(), len(set(names))))
    print("  wrote %s - now: python tools/tilt_train.py data/tilt_board.npz" % out)
    if tilts.max() - tilts.min() < 20:
        print("  WARNING: the tilt range is only %.0f degrees; a head fitted on this will not "
              "generalise past it" % (tilts.max() - tilts.min()))


if __name__ == "__main__":
    main(*sys.argv[1:])
