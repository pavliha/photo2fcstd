"""If the pipeline kept every view-and-mode candidate, how much better could it choose?"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, stats

IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))
MODES = ("plan", "profile", "revolve")
OUT = "data/hypothesis_oracle.json"


def f1_of(doc, part):
    try:
        row = SS.score_one(doc, IDEAL[part])
    except Exception:
        return None
    value = row.get("primitive_f1")
    return float(value["f1"]) if isinstance(value, dict) and value.get("wanted") else None


def one(part):
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        if not views:
            return None
        shipped = f1_of(spec_mod.assemble(views, name=part, log=lambda *a: None), part)
    except Exception:
        return None
    if shipped is None:
        return None
    candidates = {}
    for i, view in enumerate(views):
        for mode in MODES:
            try:
                doc = spec_mod.assemble([view], name=part, mode=mode, log=lambda *a: None)
            except Exception:
                continue
            got = f1_of(doc, part)
            if got is not None:
                candidates["%d_%s" % (i, mode)] = got
    if not candidates:
        return None
    return {"part": part, "shipped": shipped, "candidates": candidates}


def main():
    ids = sys.argv[1] if len(sys.argv) > 1 else "data/tune_subset.txt"
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    parts = [p for p in open(ids).read().split() if p in IDEAL and SS.trustworthy(IDEAL[p])]
    with Pool(jobs) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    json.dump(rows, open(OUT, "w"))
    shipped = np.array([r["shipped"] for r in rows])
    oracle = np.array([max(r["candidates"].values()) for r in rows])
    single = {}
    for key in sorted({k for r in rows for k in r["candidates"]}):
        vals = [r["candidates"][key] for r in rows if key in r["candidates"]]
        single[key] = (float(np.mean(vals)), len(vals))
    print("%d parts, %d candidates each at most" % (len(rows), 3 * len(MODES)))
    print("  shipped pipeline      %.4f" % shipped.mean())
    print("  best fixed candidate  %.4f  (%s)" % max((v[0], k) for k, v in single.items()))
    m, lo, hi = stats.mean_ci((oracle - shipped).tolist())
    print("  best of all candidates %.4f   headroom %+.4f [%+.4f, %+.4f]" % (oracle.mean(), m, lo, hi))
    per_view = np.array([max(v for k, v in r["candidates"].items() if k.startswith("0_")) for r in rows
                         if any(k.startswith("0_") for k in r["candidates"])])
    print("  best mode on view 1 only %.4f" % per_view.mean())


if __name__ == "__main__":
    main()
