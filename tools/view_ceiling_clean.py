"""Each part has three photos. How much is choosing the right one worth?

Estimating tilt from one silhouette is ambiguous and every hand-written criterion for it loses.
Choosing among three real photographs is a different and much better posed problem - the same
shape as the carve axis choice, which a small classifier took from 69% to 89%.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    try:
        photos = bench.photos_of(part)[:3]
        views = [analysis.view(p) for p in photos]
        per = []
        for v in views:
            try:
                doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
                per.append({"iou": SS.score_one(doc, IDEAL[part])["region_iou"],
                            "rect": v["shape"]["rectangularity"], "sol": v["shape"]["solidity"],
                            "elong": v["elongation"], "ellipse_rms": v["shape"]["ellipse_rms"],
                            "hole_frac": v["shape"]["hole_frac"], "stroke": v["shape"]["stroke_px"],
                            "nholes": len(v["shape"]["holes"]), "area": float(v["length_px"])})
            except Exception:
                per.append(None)
        shipped = SS.score_one(spec_mod.assemble(views, name=part, log=lambda *a: None),
                               IDEAL[part])["region_iou"]
        return part, {"per_view": per, "shipped": shipped}
    except Exception:
        return part, None


def main(limit=1200):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        res = {k: v for k, v in pool.map(one, parts) if v}
    json.dump(res, open(os.path.join(ROOT, "data", "view_ceiling_clean.json"), "w"))
    keys = [k for k in res if sum(1 for p in res[k]["per_view"] if p) >= 2]
    keen = [k for k in keys if 1 - SS.trivial_score(IDEAL[k]) >= 0.15]
    def col(f, ks):
        return np.mean([f(res[k]) for k in ks])
    best = lambda r: max(p["iou"] for p in r["per_view"] if p)
    worst = lambda r: min(p["iou"] for p in r["per_view"] if p)
    first = lambda r: next(p["iou"] for p in r["per_view"] if p)
    rectest = lambda r: max((p for p in r["per_view"] if p), key=lambda p: p["rect"])["iou"]
    print("%d parts, %d discriminating\n" % (len(keys), len(keen)))
    print("  %-30s %8s %8s" % ("", "all", "discriminating"))
    for name, f in (("first photo", first), ("most rectangular view", rectest),
                    ("what ships today", lambda r: r["shipped"]),
                    ("best of the three (oracle)", best), ("worst of the three", worst)):
        print("  %-30s %8.3f %8.3f" % (name, col(f, keys), col(f, keen)))
    print("\n  headroom from view choice alone: %+.3f on discriminating parts"
          % (col(best, keen) - col(lambda r: r["shipped"], keen)))


if __name__ == "__main__":
    main()
