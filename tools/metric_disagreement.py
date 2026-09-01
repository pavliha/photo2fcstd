"""Does region IoU rank drawings the way an editor would?

Every A/B in this project is judged on region IoU, and it cannot see primitive type - an arc and
the chords approximating it cover the same area - nor small holes, which carry almost no area. The
letter G scored 0.41 with correct topology while a rounded blob scored 0.81. If that inversion is
common, the acceptance criterion has been rejecting good work.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        m, i = s["counts_mine"], s["counts_ideal"]
        cm = sum(v for k, v in m.items() if k in CURVES)
        ci = sum(v for k, v in i.items() if k in CURVES)
        # structure: how much of the real sketch's shape the drawing reproduces, as an editor sees it
        loops = min(s["loops_mine"], s["loops_ideal"]) / max(s["loops_ideal"], 1)
        curves = 1.0 - abs(cm - ci) / max(ci, cm, 1)
        elems = 1.0 - abs(sum(m.values()) - sum(i.values())) / max(sum(i.values()), 1)
        return part, {"iou": s["region_iou"], "trivial": s["trivial"],
                      "loops": float(loops), "curves": float(curves), "elems": float(max(elems, 0.0)),
                      "exact": m == i, "dev": s["dev_p95"]}
    except Exception:
        return part, None


def main(limit=300):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = {k: v for k, v in pool.map(one, parts) if v}
    keen = [k for k in rows if 1 - rows[k]["trivial"] >= 0.15]
    iou = np.array([rows[k]["iou"] for k in keen])
    struct = np.array([(rows[k]["loops"] + rows[k]["curves"] + rows[k]["elems"]) / 3 for k in keen])
    exact = np.array([rows[k]["exact"] for k in keen])
    print("n=%d discriminating parts\n" % len(keen))
    print("  correlation between region IoU and structural agreement: %.2f" % np.corrcoef(iou, struct)[0, 1])
    print("  correlation between region IoU and exact primitives:     %.2f" % np.corrcoef(iou, exact)[0, 1])
    hi_iou = iou >= np.percentile(iou, 75)
    lo_struct = struct <= np.percentile(struct, 25)
    inverted = hi_iou & lo_struct
    print("\n  drawings in the top quarter by IoU and the bottom quarter by structure: %d (%.0f%%)"
          % (inverted.sum(), 100 * inverted.mean()))
    lo_iou = iou <= np.percentile(iou, 25)
    hi_struct = struct >= np.percentile(struct, 75)
    hidden = lo_iou & hi_struct
    print("  the reverse - good structure, poor IoU:                                 %d (%.0f%%)"
          % (hidden.sum(), 100 * hidden.mean()))
    names = np.array(keen)
    if inverted.any():
        print("\n  scored well, drawn badly: %s" % ", ".join(names[inverted][:6]))
    if hidden.any():
        print("  drawn well, scored badly: %s" % ", ".join(names[hidden][:6]))
    json.dump(rows, open(os.path.join(ROOT, "data", "metric_disagreement.json"), "w"))


if __name__ == "__main__":
    main()
