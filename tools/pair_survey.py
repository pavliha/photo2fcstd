"""E1: pairs of drawings of the same part whose structural terms trade off, for a person to judge.

Twenty pairs, each shipped against one knob moved - the arc gate for curves, the simplification
eps for elements, the hole floor for loops. A pair is kept only when the two drawings disagree on
which structural term is better, so every answer constrains the weights. Sides are shuffled; the
key mapping A/B back to arms is written next to the figure, not into it.
"""
import json, os, random, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))

VARIANTS = {"arcs": {"ARC_MIN_SPAN_DEG": 15.0, "ARC_MIN_SAG_FRAC": 0.03},
            "coarse": {"RUN_EPS": 0.036},
            "fine": {"RUN_EPS": 0.009},
            "holes": {"MIN_HOLE_FRAC": 0.0003}}


def apply(arm):
    from photo2fcstd import thresholds as th, trace
    defaults = {"ARC_MIN_SPAN_DEG": 40.0, "ARC_MIN_SAG_FRAC": 0.08, "MIN_HOLE_FRAC": 0.001}
    for k, v in defaults.items():
        setattr(th, k, v)
    trace.RUN_EPS = 0.01806
    for k, v in VARIANTS.get(arm, {}).items():
        if k == "RUN_EPS":
            trace.RUN_EPS = v
        else:
            setattr(th, k, v)


def one(args):
    part, arm = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    apply(arm)
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        v = SS.verdict(s)
        return part, arm, {"verdict": {k: v[k] for k in ("region_iou", "loops", "curves", "elements", "structure", "exact")},
                           "outline": doc.get("outline"), "revolve": doc.get("revolve")}
    except Exception as e:
        return part, arm, {"error": str(e)[:80]}


def trade_off(a, b):
    keys = ("loops", "curves", "elements")
    da = [a["verdict"][k] - b["verdict"][k] for k in keys]
    return any(x > 0.05 for x in da) and any(x < -0.05 for x in da)


def main(limit=250, seed=7):
    from photo2fcstd import bench, sketch_score as SS
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    jobs = [(p, a) for a in ("shipped",) + tuple(VARIANTS) for p in parts]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    by = {}
    for part, arm, v in rows:
        by.setdefault(part, {})[arm] = v
    pairs = []
    for part, arms in by.items():
        base = arms.get("shipped")
        if not base or "verdict" not in base:
            continue
        for arm in VARIANTS:
            alt = arms.get(arm)
            if not alt or "verdict" not in alt:
                continue
            if trade_off(base, alt):
                pairs.append({"part": part, "arm": arm, "shipped": base, "variant": alt})
    random.Random(seed).shuffle(pairs)
    counts = {}
    kept = []
    for p in pairs:
        if counts.get(p["arm"], 0) >= 7 or any(k["part"] == p["part"] for k in kept):
            continue
        counts[p["arm"]] = counts.get(p["arm"], 0) + 1
        kept.append(p)
        if len(kept) == 20:
            break
    out = os.path.join(ROOT, "data", "pair_survey.json")
    json.dump(kept, open(out, "w"))
    print("pairs kept: %d of %d candidates  by arm %s" % (len(kept), len(pairs), counts))
    print("wrote", out)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
