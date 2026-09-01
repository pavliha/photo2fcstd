"""Is there anything to select between? The oracle over approxPolyDP tolerances.

approxPolyDP takes one global tolerance, so it cannot be loose enough for a large outline and
tight enough for a small notch on it at the same time. Before training anything to choose, measure
whether choosing could help: score each part at several tolerances and take the best.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
EPS = (0.006, 0.010, 0.015, 0.022, 0.030)


def one(args):
    part, eps = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, trace
    trace.RUN_EPS = eps
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        return part, eps, {"iou": s["region_iou"], "trivial": s["trivial"],
                           "exact": s["counts_mine"] == s["counts_ideal"],
                           "n": sum(s["counts_mine"].values()),
                           "ideal_n": sum(s["counts_ideal"].values())}
    except Exception:
        return part, eps, None


def main(limit=200):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = pool.map(one, [(p, e) for e in EPS for p in parts])
    by = {}
    for part, eps, r in rows:
        if r:
            by.setdefault(part, {})[eps] = r
    full = [p for p, d in by.items() if len(d) == len(EPS)]
    keen = [p for p in full if 1 - by[p][EPS[0]]["trivial"] >= 0.15]
    print("n=%d parts, %d discriminating, %d tolerances each\n" % (len(full), len(keen), len(EPS)))
    print("  %-22s %8s %8s %9s" % ("tolerance", "IoU", "exact", "elements"))
    for e in EPS:
        v = [by[p][e] for p in keen]
        print("  %-22s %8.3f %7.0f%% %9.2f" % ("eps_frac %.3f" % e, np.mean([x["iou"] for x in v]),
              100 * np.mean([x["exact"] for x in v]), np.mean([x["n"] for x in v])))
    best = [max(by[p].values(), key=lambda x: x["iou"]) for p in keen]
    print("  %-22s %8.3f %7.0f%% %9.2f" % ("best per part (oracle)", np.mean([x["iou"] for x in best]),
          100 * np.mean([x["exact"] for x in best]), np.mean([x["n"] for x in best])))
    print("  %-22s %8s %8s %9.2f" % ("the real sketches", "-", "100%",
                                     np.mean([by[p][EPS[0]]["ideal_n"] for p in keen])))
    shipped = np.array([by[p][0.015]["iou"] for p in keen])
    oracle = np.array([max(by[p].values(), key=lambda x: x["iou"])["iou"] for p in keen])
    m, lo, hi = stats.mean_ci(oracle - shipped)
    print("\n  headroom from choosing the tolerance per part: %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
    agree = np.mean([max(by[p], key=lambda e: by[p][e]["iou"]) == 0.015 for p in keen])
    print("  the shipped tolerance is already best on %.0f%% of parts" % (100 * agree))
    json.dump({p: {str(e): r for e, r in d.items()} for p, d in by.items()},
              open(os.path.join(ROOT, "data", "eps_ceiling.json"), "w"))


if __name__ == "__main__":
    main()
