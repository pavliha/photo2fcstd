"""Does an element's evidence predict how wrong it is?

The tracer now records, per element, how many contour points backed it and how well they fit. That
did not help predict constraints, but the useful question is different and simpler: can we tell the
user which edges of the drawing to check? A sketch that is honest about its weak edges is worth more
than one that is marginally more accurate everywhere.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constraint_data import ideal_loops  # noqa: E402
from photo2fcstd import constraints as K, sketch_score as SS  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    from photo2fcstd import analysis, bench, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        loops = (doc.get("outline") or {}).get("loops")
        ref = ideal_loops(IDEAL[part])
        if not loops or not ref:
            return None
        mine = K.of_sketch(loops)
        partner, dist = K.align(mine["elements"], K.of_sketch(ref)["elements"])
        out = []
        for i, e in enumerate(mine["elements"]):
            s = e.get("support") or {}
            if not s:
                continue
            out.append({"err": float(dist[i]), "points": float(s.get("points", 0)),
                        "residual": float(s.get("residual", 0)),
                        "span": float(s.get("span_px", 0)),
                        "straightness": float(s.get("straightness", 1)),
                        "kind": e.get("type")})
        return out
    except Exception:
        return None


def main(limit=400):
    from photo2fcstd import bench
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = [r for chunk in pool.map(one, parts) if chunk for r in chunk]
    json.dump(rows, open(os.path.join(ROOT, "data", "support_vs_error.json"), "w"))
    err = np.array([r["err"] for r in rows])
    print("%d elements from %d parts\n" % (len(rows), len(parts)))
    print("  element error, in units of the sketch diagonal: median %.3f, p90 %.3f\n"
          % (np.median(err), np.percentile(err, 90)))
    print("  %-16s %10s  %s" % ("evidence", "corr", "median error, weakest vs strongest fifth"))
    for key, sign in (("points", -1), ("residual", 1), ("span", -1), ("straightness", -1)):
        v = np.array([r[key] for r in rows])
        c = float(np.corrcoef(v, err)[0, 1])
        order = np.argsort(sign * v)
        weak, strong = order[-len(order) // 5:], order[:len(order) // 5]
        print("  %-16s %+10.3f  %.3f vs %.3f" % (key, c, np.median(err[weak]), np.median(err[strong])))
    bad = err > np.percentile(err, 80)
    print("\n  can we flag the worst fifth of elements?")
    for key, sign in (("points", -1), ("residual", 1), ("span", -1)):
        v = np.array([r[key] for r in rows])
        cut = np.percentile(sign * v, 80)
        flagged = (sign * v) >= cut
        print("    by %-12s precision %.2f  recall %.2f  (chance %.2f)"
              % (key, bad[flagged].mean(), flagged[bad].mean(), bad.mean()))


if __name__ == "__main__":
    main()
