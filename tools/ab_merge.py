"""A/B fitting arcs across consecutive runs instead of one run at a time."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        m, i = s["counts_mine"], s["counts_ideal"]
        return part, {"iou": s["region_iou"], "trivial": s["trivial"], "exact": m == i,
                      "curve": sum(v for k, v in m.items() if k in CURVES),
                      "curve_ideal": sum(v for k, v in i.items() if k in CURVES),
                      "n": sum(m.values()), "spec": doc}
    except Exception as e:
        return part, {"error": str(e)[:60]}


def main(limit=260):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    arm = "merge" if os.environ.get("P2F_MERGE_RUNS") == "1" else "shipped"
    run = os.path.join(ROOT, "runs", "merge_%s" % arm, "out")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "list.tsv"), "w") as fh:
        for part, r in out.items():
            if "spec" in r:
                sp = os.path.join(run, part + ".spec.json")
                json.dump(r.pop("spec"), open(sp, "w"))
                fh.write("%s\t%s\n" % (sp, os.path.join(run, part + ".FCStd")))
    json.dump(out, open(os.path.join(ROOT, "data", "ab_merge_%s.json" % arm), "w"))
    print("%s: %d of %d parts produced a sketch" % (arm, sum(1 for v in out.values() if "iou" in v), len(parts)))


if __name__ == "__main__":
    main()
