"""Is polygon simplification what limits arc recovery, measured where capture noise cannot confound it?

An earlier sweep of this knob patched `outline`'s eps rather than `corner_runs`', and all four arms
returned an identical 0.639 - a null that agreed to four decimals because the knob was not wired.
This sets `trace.RUN_EPS`, the one `corner_runs` actually reads, and checks the arms differ before
believing any of it.

Run on perfect rasterised faces, so nothing here is capture.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import sketch_score as SS  # noqa: E402
import tracer_ceiling as TC  # noqa: E402

CURVED = ("arc", "circle", "ellipse", "bsplinecurve")
ARMS = {"eps/4": 0.25, "eps/2": 0.5, "shipped": 1.0, "eps x2": 2.0, "eps x4": 4.0}


def one(args):
    part, factor = args
    from photo2fcstd import analysis, spec as spec_mod, trace
    trace.RUN_EPS = 0.01806 * factor
    rec = TC.IDEAL[part]
    try:
        mask = TC.rasterise(rec)
        if mask is None or mask.sum() < 500:
            return None
        view = analysis.view_from_mask(mask.astype(np.uint8))
        loops = spec_mod.traced_outline(view)
        if not loops:
            return None
        s = SS.score_one({"outline": {"loops": loops}}, rec)
        v = SS.verdict(s)
        m = s["counts_mine"]
        v["curve_mine"] = sum(c for k, c in m.items() if k in CURVED)
        v["n_mine"] = sum(m.values())
        v["curve_ideal"] = sum(c for k, c in s["counts_ideal"].items() if k in CURVED)
        v["n_ideal"] = sum(s["counts_ideal"].values())
        return part, factor, v
    except Exception:
        return None


def main(limit=180):
    photo = json.load(open(os.path.join(ROOT, "data", "ab_arcs.json")))["shipped"]
    parts = [p for p, v in photo.items() if "n" in v and 1 - v.get("trivial", 0) >= 0.15]
    parts = [p for p in sorted(parts) if TC.IDEAL[p].get("loops")][:limit]
    jobs = [(p, f) for f in ARMS.values() for p in parts]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, jobs) if r]
    by = {}
    for part, factor, v in rows:
        by.setdefault(factor, {})[part] = v
    keen = sorted(set.intersection(*[set(d) for d in by.values()]))
    print("  n=%d parts scored in every arm, perfect input\n" % len(keen))
    print("  %-10s %8s %8s %10s %10s %10s %9s" % ("arm", "curves", "elements", "IoU", "structure", "curves t", "exact"))
    for name, f in ARMS.items():
        d = by[f]
        print("  %-10s %8.2f %8.2f %10.3f %10.3f %10.3f %8.0f%%"
              % (name, np.mean([d[p]["curve_mine"] for p in keen]), np.mean([d[p]["n_mine"] for p in keen]),
                 np.mean([d[p]["region_iou"] for p in keen]), np.mean([d[p]["structure"] for p in keen]),
                 np.mean([d[p]["curves"] for p in keen]), 100 * np.mean([d[p]["exact"] for p in keen])))
    d0 = by[1.0]
    print("  %-10s %8.2f %8.2f" % ("really", np.mean([d0[p]["curve_ideal"] for p in keen]),
                                   np.mean([d0[p]["n_ideal"] for p in keen])))
    from photo2fcstd import stats
    print()
    for name, f in ARMS.items():
        if f == 1.0:
            continue
        diff = np.array([by[f][p]["structure"] - d0[p]["structure"] for p in keen])
        m, lo, hi = stats.mean_ci(diff)
        print("  %-10s structure vs shipped %+.4f [%+.4f, %+.4f]" % (name, m, lo, hi))
    spread = max(np.mean([by[f][p]["region_iou"] for p in keen]) for f in ARMS.values()) - \
        min(np.mean([by[f][p]["region_iou"] for p in keen]) for f in ARMS.values())
    print("\n  IoU spread across arms: %.4f%s" % (spread, "  <- KNOB IS NOT WIRED" if spread < 1e-6 else ""))
    json.dump({str(k): v for k, v in by.items()}, open(os.path.join(ROOT, "data", "eps_ceiling.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
