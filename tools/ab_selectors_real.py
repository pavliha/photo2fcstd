"""Both view selectors on real views, since one of them reads fields a reconstruction cannot fake."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
SEEN = set(json.load(open(os.path.join(ROOT, "data", "view_ceiling.json"))))


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, view_model, view_rank
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        if len(views) < 3:
            return None
        per = []
        for v in views:
            doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
            per.append(SS.score_one(doc, IDEAL[part])["region_iou"])
        vm = view_model.choose(views)
        rk = view_rank.best(views)
        return {"part": part, "iou": per,
                "view_model": 0 if vm is None else int(vm),
                "view_rank": views.index(rk) if rk is not None else 0,
                "trivial": SS.trivial_score(IDEAL[part])}
    except Exception:
        return None


def main(limit=320):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p]) and p not in SEEN]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    json.dump(rows, open(os.path.join(ROOT, "data", "ab_selectors.json"), "w"))
    keen = [r for r in rows if 1 - r["trivial"] >= 0.15]
    I = np.array([r["iou"] for r in keen], float)
    idx = np.arange(len(keen))
    truth = I.argmax(1)
    print("n=%d parts, %d discriminating, none seen by view_model in training\n" % (len(rows), len(keen)))
    print("  %-22s %8s %8s" % ("selector", "agree", "IoU"))
    print("  %-22s %8.2f %8.3f" % ("chance", 1 / 3, I.mean()))
    print("  %-22s %8s %8.3f" % ("first photo", "-", I[:, 0].mean()))
    got = {}
    for name in ("view_model", "view_rank"):
        c = np.array([r[name] for r in keen], int)
        got[name] = I[idx, c]
        print("  %-22s %8.2f %8.3f" % (name, np.mean(c == truth), got[name].mean()))
    print("  %-22s %8.2f %8.3f" % ("oracle", 1.0, I.max(1).mean()))
    d = got["view_rank"] - got["view_model"]
    m, lo, hi = stats.mean_ci(d)
    print("\n  view_rank minus view_model: %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
    same = np.mean([r["view_model"] == r["view_rank"] for r in keen])
    print("  they agree with each other on %.0f%% of parts" % (100 * same))


if __name__ == "__main__":
    main()
