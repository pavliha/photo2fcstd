"""Re-sweep the thresholds now that the view choice feeds them different photos.

Every empirical constant here was jointly tuned against the old view selection, which drew from
an oblique photo 27% of the time. The shipped selector changed which images reach the tracer on
109 of 295 parts, so the distribution these constants were fitted to has moved. A/B on the same
parts, both arms regenerating specs, one constant at a time.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))

SWEEP = {
    "ARC_MIN_SAG_FRAC": (0.05, 0.08, 0.12),
    "ARC_MIN_SPAN_DEG": (25.0, 40.0, 55.0),
    "ARC_MIN_CHORD_FRAC": (0.02, 0.03, 0.045),
    "MIN_HOLE_FRAC": (0.0005, 0.001, 0.002),
    "HOLE_FRAC_VISIBLE": (0.03, 0.05, 0.08),
    "PROFILE_RECT": (0.35, 0.45, 0.55),
}


def one(args):
    part, name, value = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, thresholds as th
    if name:
        setattr(th, name, value)
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        return part, name, value, s["region_iou"], s["trivial"], s["counts_mine"] == s["counts_ideal"]
    except Exception:
        return part, name, value, None, None, None


def main(limit=160):
    from photo2fcstd import bench, sketch_score as SS, thresholds as th
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    jobs = [(p, None, None) for p in parts]
    for name, values in SWEEP.items():
        for v in values:
            if v != getattr(th, name):
                jobs += [(p, name, v) for p in parts]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    by = {}
    for part, name, value, iou, triv, exact in rows:
        if iou is None:
            continue
        by.setdefault((name, value), {})[part] = (iou, triv, exact)
    base = by.get((None, None), {})
    keen = [p for p, (i, t, e) in base.items() if 1 - t >= 0.15]
    bl = np.mean([base[p][0] for p in keen])
    print("threshold re-sweep under the shipped view choice, n=%d discriminating parts\n" % len(keen))
    print("  %-22s %8s %9s %8s %8s" % ("setting", "value", "IoU", "delta", "exact"))
    print("  %-22s %8s %9.3f %8s %7.0f%%" % ("shipped", "-", bl, "-",
                                             100 * np.mean([base[p][2] for p in keen])))
    out = []
    for (name, value), d in sorted(by.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or 0)):
        if name is None:
            continue
        common = [p for p in keen if p in d]
        v = np.mean([d[p][0] for p in common])
        b = np.mean([base[p][0] for p in common])
        out.append((v - b, name, value, v, np.mean([d[p][2] for p in common])))
        print("  %-22s %8s %9.3f %+8.4f %7.0f%%" % (name, value, v, v - b, 100 * out[-1][4]))
    out.sort(reverse=True)
    print("\n  best change: %s = %s, %+.4f" % (out[0][1], out[0][2], out[0][0]) if out else "")
    json.dump([[a, b, c, d, e] for a, b, c, d, e in out],
              open(os.path.join(ROOT, "data", "threshold_resweep.json"), "w"))


if __name__ == "__main__":
    main()
