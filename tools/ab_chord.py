"""Is the arc gate rejecting arcs because they are small, or because the *part* is big?

`trace.py` asks `chord > ARC_MIN_CHORD_FRAC * length_px` before a run may be fitted as an arc - a
fraction of the whole part's extent. On a part with 28 curves every arc is a small share of it and
is thrown out before any fitting happens, which is what makes recovery fall from 99% on parts with
one curve to 23% on parts with sixteen.

The gate's job is to ask whether a run is big enough for its curvature to be measurable, and that
is a question about pixels, not about the part. Arms below replace the relative test with an
absolute one and with looser relative ones, on perfect rasterised faces so nothing here is capture.

**Precision is the thing to watch.** The earlier gate loosening fired on 54% of parts whose sketch
has no curve at all; every arm reports the false-curve rate on the parts that really have none.
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
ARMS = {"shipped": 0, "merge 2": 2, "merge 3": 3, "merge 4": 4}


def one(args):
    part, arm = args
    from photo2fcstd import analysis, spec as spec_mod, trace
    trace.MERGE_RUNS = ARMS[arm]
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
        m, i = s["counts_mine"], s["counts_ideal"]
        v["curve_mine"] = sum(c for k, c in m.items() if k in CURVED)
        v["curve_ideal"] = sum(c for k, c in i.items() if k in CURVED)
        v["n_mine"] = sum(m.values())
        return part, arm, v
    except Exception:
        return None


def main(limit=420):
    IDEAL = TC.IDEAL
    trusted = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p]) and IDEAL[p].get("loops")]
    other = [p for p in sorted(IDEAL) if p not in set(trusted) and IDEAL[p].get("loops")]
    parts = (trusted + other)[:limit]
    jobs = [(p, a) for a in ARMS for p in parts]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, jobs) if r]
    by = {}
    for part, arm, v in rows:
        by.setdefault(arm, {})[part] = v
    keen = sorted(set.intersection(*[set(d) for d in by.values()]))
    base = by["shipped"]
    zero = [p for p in keen if base[p]["curve_ideal"] == 0]
    curvy = [p for p in keen if base[p]["curve_ideal"] >= 4]
    print("  n=%d parts in every arm; %d have no curve at all, %d have four or more\n"
          % (len(keen), len(zero), len(curvy)))
    print("  %-11s %9s %9s %11s %10s %10s %9s"
          % ("arm", "curves", "recovered", "false/part", "structure", "elements", "exact"))
    for arm in ARMS:
        d = by[arm]
        cm = np.mean([d[p]["curve_mine"] for p in curvy])
        ci = np.mean([d[p]["curve_ideal"] for p in curvy])
        print("  %-11s %9.2f %10.0f%% %11.2f %10.3f %10.2f %8.0f%%"
              % (arm, cm, 100 * cm / max(ci, 1e-9),
                 np.mean([d[p]["curve_mine"] for p in zero]) if zero else 0.0,
                 np.mean([d[p]["structure"] for p in keen]),
                 np.mean([d[p]["n_mine"] for p in keen]),
                 100 * np.mean([d[p]["exact"] for p in keen])))
    print("  %-11s %9.2f" % ("really", np.mean([base[p]["curve_ideal"] for p in curvy])))
    from photo2fcstd import stats
    print()
    for arm in ARMS:
        if arm == "shipped":
            continue
        d = np.array([by[arm][p]["structure"] - base[p]["structure"] for p in keen])
        m, lo, hi = stats.mean_ci(d)
        flag = "  <- better" if lo > 0 else ("  <- worse" if hi < 0 else "")
        print("  %-11s structure vs shipped %+.4f [%+.4f, %+.4f]%s" % (arm, m, lo, hi, flag))
    json.dump({a: {p: v for p, v in d.items()} for a, d in by.items()},
              open(os.path.join(ROOT, "data", "ab_chord.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
