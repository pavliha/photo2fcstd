"""Which of the four arc conditions actually refuses each run?

A6 showed the chord gate explains about 5 of the 65 missing points, and A8's decomposition puts
39% of lost arcs in a bucket that clears the sweep gate both whole and per piece - so something
else is refusing them. This runs the production arithmetic from `trace.elements` over the same runs
and tallies which condition fails first, on perfect rasterised faces.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import sketch_score as SS, thresholds as th  # noqa: E402
import tracer_ceiling as TC  # noqa: E402


def verdicts_for(raw, length_px):
    from photo2fcstd import trace
    repeated = trace.boundary_period(raw)
    runs = trace.corner_runs(raw, trace.REPEATED_RUN_EPS if repeated else None)
    out = []
    for run in runs:
        chord = float(np.hypot(*(run[-1] - run[0])))
        if len(run) < th.ARC_MIN_POINTS:
            out.append("too few points")
            continue
        if chord <= th.ARC_MIN_CHORD_FRAC * length_px:
            out.append("chord too short for the part")
            continue
        cx, cy, r, rel, _ = trace.fit_circle(run)
        span = trace.arc_span(run, cx, cy)
        cv = run[-1] - run[0]
        sag = float(np.max(np.abs(cv[0] * (run[:, 1] - run[0][1]) - cv[1] * (run[:, 0] - run[0][0]))
                           / max(chord, 1e-9)))
        if not rel * r < max(th.ARC_FIT_TOL * r, 1.2):
            out.append("circle fit too loose")
        elif not th.ARC_MIN_SPAN_DEG < abs(span):
            out.append("sweep under 40 deg")
        elif not abs(span) < th.ARC_MAX_SPAN_DEG:
            out.append("sweep over 350 deg")
        elif not sag > th.ARC_MIN_SAG_FRAC * chord:
            out.append("too flat (sagitta)")
        else:
            out.append("ACCEPTED as an arc")
    return out


def one(part):
    from photo2fcstd import analysis
    rec = TC.IDEAL[part]
    try:
        mask = TC.rasterise(rec)
        if mask is None or mask.sum() < 500:
            return []
        view = analysis.view_from_mask(mask.astype(np.uint8))
        raw = np.array(view["shape"]["raw"], float)
        return verdicts_for(raw, view["length_px"])
    except Exception:
        return []


def main(limit=420):
    IDEAL = TC.IDEAL
    trusted = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p]) and IDEAL[p].get("loops")]
    other = [p for p in sorted(IDEAL) if p not in set(trusted) and IDEAL[p].get("loops")]
    parts = (trusted + other)[:limit]
    with Pool(6) as pool:
        rows = [v for vs in pool.map(one, parts) for v in vs]
    total = len(rows)
    print("  %d contour runs over %d parts, perfect input\n" % (total, len(parts)))
    counts = {}
    for v in rows:
        counts[v] = counts.get(v, 0) + 1
    print("  %-32s %7s %8s" % ("outcome", "runs", "share"))
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        print("  %-32s %7d %7.0f%%" % (name, n, 100 * n / max(total, 1)))
    rejected = total - counts.get("ACCEPTED as an arc", 0)
    print("\n  of the %d runs not made into arcs, the leading cause is %s (%.0f%% of rejections)"
          % (rejected, max(((k, v) for k, v in counts.items() if k != "ACCEPTED as an arc"),
                           key=lambda x: x[1])[0],
             100 * max(v for k, v in counts.items() if k != "ACCEPTED as an arc") / max(rejected, 1)))
    json.dump(counts, open(os.path.join(ROOT, "data", "gate_census.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
