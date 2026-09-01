"""When every photo gives a bad sketch, is the tracing losing it or was it never there?"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import bench, sketch_score as SS, stats
from photo2fcstd.trace import outline, segment_photo, upright_mask

IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))


def raw_doc(mask):
    poly, shape = outline(mask)
    loops = [shape["raw"]] + shape["raw_holes"]
    out = []
    for pts in loops:
        pts = np.asarray(pts, float)
        out.append({"type": "loop", "elements": [
            {"type": "line", "p0": pts[i].tolist(), "p1": pts[(i + 1) % len(pts)].tolist()}
            for i in range(len(pts))]})
    return {"mode": "plan", "outline": {"loops": out}}


def best_raw(part):
    best = 0.0
    for path in bench.photos_of(part)[:3]:
        try:
            mask, _ = upright_mask(segment_photo(path))
            got = SS.score_one(raw_doc(mask), IDEAL[part])["region_iou"]
        except Exception:
            continue
        best = max(best, float(got or 0.0))
    return best


def main(limit=90):
    shipped = bench.sketch_scores("runs/final")
    bad = [p for p, r in sorted(shipped.items())
           if r.get("trustworthy") and r.get("region_iou") is not None and r["region_iou"] < 0.4]
    bad = bad[:limit]
    rows = []
    for part in bad:
        if part not in IDEAL:
            continue
        rows.append((part, float(shipped[part]["region_iou"]), best_raw(part)))
    if not rows:
        print("no parts")
        return
    ship = np.array([r[1] for r in rows])
    raw = np.array([r[2] for r in rows])
    m, lo, hi = stats.mean_ci((raw - ship).tolist())
    print("parts whose shipped sketch scores under 0.4: %d examined" % len(rows))
    print("  shipped sketch      %.3f" % ship.mean())
    print("  best raw trace      %.3f" % raw.mean())
    print("  raw minus shipped  %+.3f [%+.3f, %+.3f]" % (m, lo, hi))
    print("  raw trace also under 0.4: %.0f%% - the shape is not in any photo"
          % (100 * np.mean(raw < 0.4)))
    print("  raw above 0.6 while shipped is under 0.4: %.0f%% - regularisation loses it"
          % (100 * np.mean((raw > 0.6) & (ship < 0.4))))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 90)
