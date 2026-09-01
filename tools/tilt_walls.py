"""What tilt actually costs, split into the part a homography can undo and the part it cannot.

Knowing a photo's tilt is only worth something if the tilt can then be corrected. It cannot.
Rendering each part twice, square on and at 15 degrees, as the full solid silhouette and as the
bare base face, separates the two effects: projective distortion of the face, which a homography
removes, and the side walls coming into view, which no 2D warp can remove because the information
is simply not in the image.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, cv2, trimesh
import tilt_close as T
from photo2fcstd.bench import truth_of
from photo2fcstd import sketch_score as SS, stats
from multiprocessing import Pool


def face_mask(rec, c, view):
    img = np.zeros((T.H, T.W), np.uint8)
    for loop in rec["loops"]:
        pts = [e[k] for e in loop for k in ("p0", "p1") if k in e] or [e["c"] for e in loop if "c" in e]
        if len(pts) < 3:
            continue
        cv2.fillPoly(img, [np.round(T.C.project(np.array(pts, float) - c, view)).astype(np.int32)], 1)
    return img > 0


def one(part):
    rec = T.IDEAL[part]
    n = T.face_normal(rec)
    try:
        m = trimesh.load(truth_of(part))
        assert len(m.faces)
    except Exception:
        return None
    c = m.bounds.mean(0)
    m.apply_translation(-c)
    rad = float(np.max(m.extents)) * 4
    out = {}
    def sc(msk):
        if msk.sum() < 50:
            return np.nan
        try:
            s = T.score_mask(msk, rec)
        except Exception:
            return np.nan
        return s["region_iou"] if s else np.nan
    for tilt in (0.0, 15.0):
        v = T.view_at(n, tilt, 40.0, rad)
        ncam = v["R"] @ n
        _, sol = T.shaded(m, v)
        fac = face_mask(rec, c, v)
        out[tilt] = (sc(sol), sc(T.rectify(sol, ncam, T.K_of())), sc(fac), sc(T.rectify(fac, ncam, T.K_of())))
    return part, out


if __name__ == "__main__":
    parts = [p for p in sorted(T.IDEAL) if SS.trustworthy(T.IDEAL[p]) and T.face_normal(T.IDEAL[p]) is not None][:120]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    ok = [r for _, r in rows if not any(np.isnan(v) for t in r for v in r[t])]
    A = np.array([[r[0][0], r[0][2], r[15.0][0], r[15.0][1], r[15.0][2], r[15.0][3]] for r in ok])
    print("  n=%d parts with a real base face and a mesh\n" % len(A))
    print("  %-34s %8s" % ("", "sketch IoU"))
    print("  %-34s %8.3f" % ("solid silhouette, square on", A[:, 0].mean()))
    print("  %-34s %8.3f" % ("solid silhouette, 15 deg", A[:, 2].mean()))
    print("  %-34s %8.3f" % ("  ... rectified by the true normal", A[:, 3].mean()))
    print()
    print("  %-34s %8.3f" % ("bare base face, square on", A[:, 1].mean()))
    print("  %-34s %8.3f" % ("bare base face, 15 deg", A[:, 4].mean()))
    print("  %-34s %8.3f" % ("  ... rectified by the true normal", A[:, 5].mean()))
    print()
    m, lo, hi = stats.mean_ci(A[:, 3] - A[:, 2])
    print("  rectifying the solid at 15 deg:  %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
    m, lo, hi = stats.mean_ci(A[:, 4] - A[:, 1])
    print("  what 15 deg costs the bare face: %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
    m, lo, hi = stats.mean_ci(A[:, 2] - A[:, 0])
    print("  what 15 deg costs the solid:     %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
