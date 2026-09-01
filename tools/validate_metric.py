"""What does a perfect sketch, extruded by the true depth, score against the truth solid?"""
import json
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon

sys.path.insert(0, "src")
from photo2fcstd import score, stats, sketch_score as SS
from photo2fcstd.bench import truth_of

IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))


def polygon_of(loop):
    pts = [q for e in loop for q in e.get("xy", [])]
    if len(pts) < 4:
        return None
    poly = Polygon(pts)
    return poly if poly.is_valid and poly.area > 0 else poly.buffer(0)


def solid_of(part):
    rec = IDEAL.get(part)
    if not rec or not rec.get("loops") or not rec.get("depth"):
        return None
    outer = polygon_of(rec["loops"][0])
    if outer is None or outer.is_empty:
        return None
    holes = [polygon_of(l) for l in rec["loops"][1:]]
    for h in holes:
        if h is not None and not h.is_empty:
            try:
                outer = outer.difference(h)
            except Exception:
                pass
    try:
        return trimesh.creation.extrude_polygon(outer, height=max(float(rec["depth"]), 1e-3))
    except Exception:
        return None


def main(limit=120):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])][:limit]
    rows = []
    for part in parts:
        mesh = solid_of(part)
        if mesh is None:
            continue
        try:
            value, _ = score.best_iou(truth_of(part), mesh)
        except Exception:
            continue
        rows.append((part, float(value)))
    vals = [v for _, v in rows]
    m, lo, hi = stats.mean_ci(vals)
    print("perfect sketch + true depth, scored by the bench metric on %d parts" % len(rows))
    print("  mean %.3f [%.3f, %.3f]   median %.3f" % (m, lo, hi, np.median(vals)))
    for edge in (0.95, 0.9, 0.8, 0.6):
        print("  scores below %.2f: %3d parts (%.0f%%)"
              % (edge, sum(1 for v in vals if v < edge), 100 * np.mean([v < edge for v in vals])))
    worst = sorted(rows, key=lambda r: r[1])[:6]
    print("  worst:", ", ".join("%s %.2f" % r for r in worst))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
