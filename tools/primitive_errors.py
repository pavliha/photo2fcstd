"""What stops a drawing from reproducing the real sketch's primitives?

Sketch IoU is 0.602 but only 26% of parts come out with the right primitives, and for anyone who
has to edit the result that second number is the one that matters. Region IoU cannot see primitive
type at all - an arc and the chords approximating it cover the same area - so this counts the
errors directly instead.
"""
import json, os, sys
from collections import Counter
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
        mine, ideal = s["counts_mine"], s["counts_ideal"]
        cm = sum(v for k, v in mine.items() if k in CURVES)
        ci = sum(v for k, v in ideal.items() if k in CURVES)
        return {"part": part, "iou": s["region_iou"], "trivial": s["trivial"],
                "loops_mine": s["loops_mine"], "loops_ideal": s["loops_ideal"],
                "n_mine": sum(mine.values()), "n_ideal": sum(ideal.values()),
                "curve_mine": cm, "curve_ideal": ci,
                "line_mine": mine.get("line", 0), "line_ideal": ideal.get("line", 0),
                "exact": mine == ideal, "mode": doc["mode"]}
    except Exception:
        return None


def main(limit=400):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    keen = [r for r in rows if 1 - r["trivial"] >= 0.15]
    json.dump(rows, open(os.path.join(ROOT, "data", "primitive_errors.json"), "w"))
    n = len(keen)
    print("n=%d discriminating parts, %.0f%% reproduce the exact primitives\n"
          % (n, 100 * np.mean([r["exact"] for r in keen])))

    def frac(f):
        return 100 * np.mean([f(r) for r in keen])
    print("  %-38s %6s" % ("what goes wrong", "parts"))
    print("  %-38s %5.0f%%" % ("too few loops (missed a hole)", frac(lambda r: r["loops_mine"] < r["loops_ideal"])))
    print("  %-38s %5.0f%%" % ("too many loops", frac(lambda r: r["loops_mine"] > r["loops_ideal"])))
    print("  %-38s %5.0f%%" % ("right loop count", frac(lambda r: r["loops_mine"] == r["loops_ideal"])))
    print("  %-38s %5.0f%%" % ("  ... and exact primitives", frac(lambda r: r["loops_mine"] == r["loops_ideal"] and r["exact"])))
    print()
    print("  %-38s %5.0f%%" % ("too few curves (arc drawn as lines)", frac(lambda r: r["curve_mine"] < r["curve_ideal"])))
    print("  %-38s %5.0f%%" % ("too many curves", frac(lambda r: r["curve_mine"] > r["curve_ideal"])))
    print("  %-38s %5.0f%%" % ("right curve count", frac(lambda r: r["curve_mine"] == r["curve_ideal"])))
    print()
    el = np.array([[r["n_mine"], r["n_ideal"]] for r in keen], float)
    lp = np.array([[r["loops_mine"], r["loops_ideal"]] for r in keen], float)
    cv = np.array([[r["curve_mine"], r["curve_ideal"]] for r in keen], float)
    print("  %-24s %8s %8s" % ("", "ours", "ideal"))
    for name, a in (("elements per sketch", el), ("loops per sketch", lp), ("curves per sketch", cv)):
        print("  %-24s %8.2f %8.2f" % (name, a[:, 0].mean(), a[:, 1].mean()))
    right = [r for r in keen if r["loops_mine"] == r["loops_ideal"]]
    if right:
        print("\n  among the %d with the right loop count, %.0f%% still miss on primitive type"
              % (len(right), 100 * np.mean([not r["exact"] for r in right])))
    miss = [r for r in keen if r["loops_mine"] < r["loops_ideal"]]
    if miss:
        print("  parts missing loops are missing %.1f of them on average"
              % np.mean([r["loops_ideal"] - r["loops_mine"] for r in miss]))


if __name__ == "__main__":
    main()
