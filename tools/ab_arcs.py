"""Re-run the arc gates against the metric that can actually see them.

Loosening the gates was rejected because it moved region IoU by nothing. Region IoU cannot see
primitive type - an arc and the chords approximating it cover the same area - so that verdict was
taken on a measurement structurally incapable of showing the benefit. We draw 1.31 curves per
sketch against a real 4.95, and 48% of parts draw an arc as lines. The costs it was also rejected
for, null solids, are real and get measured here through FreeCAD alongside.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")
ARMS = {
    "shipped": {},
    "span15_sag03": {"ARC_MIN_SPAN_DEG": 15.0, "ARC_MIN_SAG_FRAC": 0.03},
    "span10_sag02": {"ARC_MIN_SPAN_DEG": 10.0, "ARC_MIN_SAG_FRAC": 0.02},
    "span15_sag03_pts6": {"ARC_MIN_SPAN_DEG": 15.0, "ARC_MIN_SAG_FRAC": 0.03, "ARC_MIN_POINTS": 6},
}


def one(args):
    part, arm = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, thresholds as th
    for k, v in ARMS[arm].items():
        setattr(th, k, v)
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        m, i = s["counts_mine"], s["counts_ideal"]
        return part, arm, {"iou": s["region_iou"], "trivial": s["trivial"], "exact": m == i,
                           "curve_mine": sum(v for k, v in m.items() if k in CURVES),
                           "curve_ideal": sum(v for k, v in i.items() if k in CURVES),
                           "n": sum(m.values()), "spec": doc}
    except Exception as e:
        return part, arm, {"error": str(e)[:60]}


def main(limit=220):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    jobs = [(p, a) for a in ARMS for p in parts]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    by = {}
    for part, arm, r in rows:
        by.setdefault(arm, {})[part] = r
    for arm, d in by.items():
        out = os.path.join(ROOT, "runs", "arcs_%s" % arm, "out")
        os.makedirs(out, exist_ok=True)
        for part, r in d.items():
            if "spec" in r:
                json.dump(r["spec"], open(os.path.join(out, part + ".spec.json"), "w"))
        for r in d.values():
            r.pop("spec", None)
    json.dump(by, open(os.path.join(ROOT, "data", "ab_arcs.json"), "w"))
    keen = [p for p in parts if all("iou" in by[a].get(p, {}) for a in ARMS)
            and 1 - by["shipped"][p]["trivial"] >= 0.15]
    print("n=%d discriminating parts, all arms scored on the same parts\n" % len(keen))
    print("  %-20s %8s %8s %8s %9s" % ("arm", "IoU", "exact", "curves", "elements"))
    for arm in ARMS:
        d = by[arm]
        print("  %-20s %8.3f %7.0f%% %8.2f %9.2f"
              % (arm, np.mean([d[p]["iou"] for p in keen]),
                 100 * np.mean([d[p]["exact"] for p in keen]),
                 np.mean([d[p]["curve_mine"] for p in keen]),
                 np.mean([d[p]["n"] for p in keen])))
    print("  %-20s %8s %8s %8.2f" % ("the real sketches", "-", "100%",
                                     np.mean([by["shipped"][p]["curve_ideal"] for p in keen])))
    for arm in ARMS:
        if arm == "shipped":
            continue
        d = np.array([by[arm][p]["iou"] - by["shipped"][p]["iou"] for p in keen])
        from photo2fcstd import stats
        m, lo, hi = stats.mean_ci(d)
        print("\n  %s IoU vs shipped: %+.4f [%+.4f, %+.4f]" % (arm, m, lo, hi))
        print("  %s exact vs shipped: %+.0f points" % (arm,
              100 * (np.mean([by[arm][p]["exact"] for p in keen]) - np.mean([by["shipped"][p]["exact"] for p in keen]))))


if __name__ == "__main__":
    main()
