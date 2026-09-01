"""How much is left on the table by choosing the wrong mode?

Mode choice is a selection among candidates the geometry already produced, which is the one shape
of problem a learned component has won at here twice. So before proposing anything, measure the
headroom: score every allowed mode for every part and compare what was chosen against the best
that was available.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
MODES = ("plan", "profile", "revolve")


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    out = {}
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
    except Exception as e:
        return part, {"error": str(e)[:60]}
    for mode in (None,) + MODES:
        try:
            doc = spec_mod.assemble(views, name=part, mode=mode, log=lambda *a: None)
            v = SS.verdict(SS.score_one(doc, IDEAL[part]))
            v["mode"] = doc["mode"]
            out["chosen" if mode is None else mode] = v
        except Exception as e:
            out["chosen" if mode is None else mode] = {"error": str(e)[:60]}
    return part, out


def main(limit=200):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    with Pool(6) as pool:
        rows = dict(pool.map(one, parts))
    json.dump(rows, open(os.path.join(ROOT, "data", "mode_oracle.json"), "w"))
    keen = [p for p, r in rows.items()
            if "region_iou" in r.get("chosen", {}) and r["chosen"]["discriminating"]
            and any("region_iou" in r.get(m, {}) for m in MODES)]
    print("  n=%d discriminating parts\n" % len(keen))

    def best(p, key):
        cand = [rows[p][m][key] for m in MODES if "region_iou" in rows[p].get(m, {})]
        return max(cand) if cand else rows[p]["chosen"][key]

    for key in ("region_iou", "structure"):
        got = np.array([rows[p]["chosen"][key] for p in keen])
        top = np.array([best(p, key) for p in keen])
        m, lo, hi = stats.mean_ci(top - got)
        print("  %-12s chosen %.3f, best available %.3f, headroom %+.4f [%+.4f, %+.4f]"
              % (key, got.mean(), top.mean(), m, lo, hi))
    agree = np.mean([rows[p]["chosen"]["region_iou"] >= best(p, "region_iou") - 1e-9 for p in keen])
    print("\n  the chosen mode was already the best for %.0f%% of parts" % (100 * agree))
    print("\n  %-10s %6s %8s %10s" % ("mode", "n", "IoU", "structure"))
    for m in MODES:
        d = [rows[p][m] for p in keen if "region_iou" in rows[p].get(m, {})]
        if d:
            print("  %-10s %6d %8.3f %10.3f"
                  % (m, len(d), np.mean([x["region_iou"] for x in d]), np.mean([x["structure"] for x in d])))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
