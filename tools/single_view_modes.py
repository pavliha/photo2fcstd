"""Scoring one view at a time is how the view-selection labels were made. Is it sound?"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    out = []
    try:
        for p in bench.photos_of(part)[:3]:
            v = analysis.view(p)
            doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
            s = SS.score_one(doc, IDEAL[part])
            out.append({"mode": doc["mode"], "drew": bool(doc.get("outline") or doc.get("revolve")),
                        "iou": s["region_iou"]})
        return part, out
    except Exception:
        return part, None


def main(limit=260):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        res = {k: v for k, v in pool.map(one, parts) if v and len(v) == 3}
    tag = "on" if os.environ.get("P2F_MODE_PIXELS", "1") == "1" else "off"
    json.dump(res, open(os.path.join(ROOT, "data", "single_view_modes_%s.json" % tag), "w"))
    flat = [r for v in res.values() for r in v]
    modes = Counter(r["mode"] for r in flat)
    blank = [r for r in flat if not r["drew"]]
    print("mode model %s: %d parts, %d single-view specs" % (tag, len(res), len(flat)))
    print("  modes: %s" % dict(modes))
    print("  drew nothing: %d (%.0f%% of views)" % (len(blank), 100 * len(blank) / max(len(flat), 1)))
    hurt = sum(1 for v in res.values()
               if max(r["iou"] for r in v) > 0 and any(not r["drew"] for r in v))
    print("  parts where at least one view was blanked: %d (%.0f%%)" % (hurt, 100 * hurt / max(len(res), 1)))
    best_blank = sum(1 for v in res.values()
                     if not v[int(np.argmax([r["iou"] for r in v]))]["drew"])
    print("  parts whose best-scoring view drew nothing: %d" % best_blank)


if __name__ == "__main__":
    main()
