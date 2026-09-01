"""Does honouring the allowed-mode list change full three-view specs?"""
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
        return part, {"iou": s["region_iou"], "trivial": s["trivial"], "mode": doc["mode"],
                      "exact": s["counts_mine"] == s["counts_ideal"],
                      "drew": bool(doc.get("outline") or doc.get("revolve"))}
    except Exception as e:
        return part, {"error": str(e)[:60]}


def main(limit=300):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    tag = os.environ.get("P2F_TAG", "fixed")
    json.dump(out, open(os.path.join(ROOT, "data", "ab_modefix_%s.json" % tag), "w"))
    ok = [v for v in out.values() if "iou" in v]
    print("%s: %d parts, %d drew, mean IoU %.3f" % (tag, len(ok), sum(v["drew"] for v in ok),
                                                    np.mean([v["iou"] for v in ok])))


if __name__ == "__main__":
    main()
