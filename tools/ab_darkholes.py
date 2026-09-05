import json, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))

def one(args):
    part, on = args
    os.environ["P2F_RECOVER_DARK"] = "1" if on else "0"
    os.environ["P2F_MASK_NOCACHE"] = "1"
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, trace
    trace.RECOVER_DARK = on
    try:
        # bypass cache: segment fresh both arms
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        return part, on, float((s.get("primitive_f1") or {}).get("f1", 0.0))
    except Exception:
        return part, on, None

def main(limit=160):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    with Pool(6) as pool:
        rows = pool.map(one, [(p, o) for o in (False, True) for p in parts])
    by = {}
    for p, o, f in rows:
        by.setdefault(o, {})[p] = f
    keen = [p for p in parts if by[False].get(p) is not None and by[True].get(p) is not None]
    d = np.array([by[True][p] - by[False][p] for p in keen])
    m, lo, hi = stats.mean_ci(d)
    print("DARKHOLES n=%d delta %+.4f [%+.4f, %+.4f] changed %d" % (len(keen), m, lo, hi, int((d != 0).sum())))

if __name__ == "__main__":
    main()
