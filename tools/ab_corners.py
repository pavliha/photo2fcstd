"""A/B the learned corners against approxPolyDP, end to end on sketch IoU with its baseline."""
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
                      "loops": s["loops_mine"], "ideal_loops": s["loops_ideal"],
                      "exact": s["counts_mine"] == s["counts_ideal"],
                      "curve": s["curve_frac_mine"]}
    except Exception:
        return part, None


def main(limit=200):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    arm = ("learned" if os.environ.get("P2F_LEARNED_CORNERS") == "1"
           else "filtered" if os.environ.get("P2F_FILTERED_CORNERS") == "1" else "approxPolyDP")
    json.dump(out, open(os.path.join(ROOT, "data", "ab_corners_%s.json" % arm), "w"))
    ok = {k: v for k, v in out.items() if v}
    print("%s: %d of %d parts produced a sketch" % (arm, len(ok), len(parts)))


if __name__ == "__main__":
    main()
