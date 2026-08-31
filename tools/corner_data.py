import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import synth  # noqa: E402
from photo2fcstd.sketch_score import trustworthy  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
PER_PART = 8


def one(args):
    part, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(PER_PART):
        try:
            s = synth.sample(IDEAL[part], rng)
        except Exception:
            s = None
        if s and (s["dist"] < 8.0).mean() > 0.85 and len(s["corners"]):
            out.append({"part": part, "contour": s["contour"].round(2).tolist(),
                        "heat": s["heat"].round(4).tolist(),
                        "corners": s["corners"].round(2).tolist()})
    return out


def main():
    parts = [k for k in sorted(IDEAL) if trustworthy(IDEAL[k])
             and max(len(lp) for lp in IDEAL[k]["loops"]) >= 2]
    with Pool(6) as pool:
        rows = [r for chunk in pool.map(one, [(p, i) for i, p in enumerate(parts)]) for r in chunk]
    np.save(os.path.join(ROOT, "data", "corner_rows.npy"), np.array(rows, dtype=object),
            allow_pickle=True)
    print("%d samples from %d parts" % (len(rows), len({r["part"] for r in rows})))
    print("mean corners per sample: %.1f" % np.mean([len(r["corners"]) for r in rows]))


if __name__ == "__main__":
    main()
