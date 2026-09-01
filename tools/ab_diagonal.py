"""A/B snapping to multiples of 45 against the shipped horizontal/vertical pass."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        return part, {"iou": s["region_iou"], "trivial": s["trivial"],
                      "exact": s["counts_mine"] == s["counts_ideal"],
                      "n": sum(s["counts_mine"].values())}
    except Exception as e:
        return part, {"error": str(e)[:50]}


def main(limit=240):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    tag = os.environ.get("P2F_DIAGONAL_TOL", "0")
    json.dump(out, open(os.path.join(ROOT, "data", "ab_diag_%s.json" % tag), "w"))
    print("tol=%s: %d of %d produced a sketch" % (tag, sum(1 for v in out.values() if "iou" in v), len(parts)))


if __name__ == "__main__":
    main()
