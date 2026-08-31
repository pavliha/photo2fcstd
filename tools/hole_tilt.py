"""When a part has a round hole, its image is an ellipse whose squash is the tilt. Does that help?

Silhouette statistics cannot say which photo was taken squarest - measured three ways. A round
hole is different in kind: it is a measurement of the tilt rather than a correlate of it. Only
12% of parts have one, so this asks whether the idea works at all where it is available.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def roundest_hole(view):
    """Aspect of the best-fitting ellipse among the holes: 1.0 means square on."""
    best = None
    for h in view["shape"].get("raw_holes", []):
        p = np.asarray(h, np.float32)
        if len(p) < 12:
            continue
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(p)
        a, b = max(MA, ma) / 2, min(MA, ma) / 2
        if b < 4:
            continue
        t = np.radians(ang)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = (p - [cx, cy]) @ R.T
        rms = float(np.sqrt(np.mean((np.hypot(q[:, 0] / max(ma / 2, 1e-6),
                                              q[:, 1] / max(MA / 2, 1e-6)) - 1) ** 2)))
        if rms < 0.10 and (best is None or a > best[1]):
            best = (b / a, a)
    return best


def one(part):
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        rows = []
        for v in views:
            r = roundest_hole(v)
            if r is None:
                return part, None
            doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
            rows.append({"aspect": r[0], "iou": SS.score_one(doc, IDEAL[part])["region_iou"]})
        return part, rows
    except Exception:
        return part, None


def main(limit=250):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        res = {k: v for k, v in pool.map(one, parts) if v and len(v) == 3}
    json.dump(res, open(os.path.join(ROOT, "data", "hole_tilt.json"), "w"))
    keen = [k for k in res if 1 - SS.trivial_score(IDEAL[k]) >= 0.15]
    if not keen:
        print("no discriminating parts have a round hole in all three photos")
        return
    A = np.array([[v["iou"] for v in res[k]] for k in keen])
    asp = np.array([[v["aspect"] for v in res[k]] for k in keen])
    pick = asp.argmax(axis=1)
    print("%d parts have a measurable round hole in all three photos\n" % len(keen))
    print("  %-28s %8s" % ("chooser", "IoU"))
    print("  %-28s %8.3f" % ("first photo", A[:, 0].mean()))
    print("  %-28s %8.3f" % ("roundest hole", A[np.arange(len(A)), pick].mean()))
    print("  %-28s %8.3f" % ("best of three (oracle)", A.max(axis=1).mean()))
    print("  %-28s %8.3f" % ("random view", A.mean()))
    print("\n  the roundest hole is the best view %.0f%% of the time (chance %.0f%%)"
          % (100 * np.mean(pick == A.argmax(axis=1)), 100 / 3))
    print("  implied tilt of the chosen view: median %.1f deg"
          % np.median(np.degrees(np.arccos(np.clip(asp[np.arange(len(asp)), pick], 0, 1)))))


if __name__ == "__main__":
    main()
