import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))

def one(args):
    part, veto = args
    from photo2fcstd import analysis, bench, modes, sketch_score as SS, spec as spec_mod
    modes.RECT_VETO_FRAC = veto
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        return part, veto, float((s.get("primitive_f1") or {}).get("f1", 0.0))
    except Exception:
        return part, veto, None

def main(limit=200):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    jobs = [(p, v) for v in (0.0, 0.6) for p in parts]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    by = {}
    for p, v, f in rows:
        by.setdefault(v, {})[p] = f
    keen = [p for p in parts if by[0.0].get(p) is not None and by[0.6].get(p) is not None]
    d = np.array([by[0.6][p] - by[0.0][p] for p in keen])
    m, lo, hi = stats.mean_ci(d)
    print("RECTVETO n=%d delta %+.4f [%+.4f, %+.4f] changed %d" % (len(keen), m, lo, hi, int((d != 0).sum())))

if __name__ == "__main__":
    main()
