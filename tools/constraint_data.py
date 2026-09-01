"""Build the constraint training set, and report whether its labels are trustworthy."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def ideal_loops(record):
    out = []
    for lp in record["loops"]:
        els = []
        for e in lp:
            xy = np.asarray(e.get("xy", []), float)
            if len(xy) < 2:
                continue
            if e.get("type", "").lower() == "line":
                els.append({"type": "line", "p0": xy[0].tolist(), "p1": xy[-1].tolist()})
            else:
                c = xy.mean(axis=0)
                els.append({"type": "arc", "p0": xy[0].tolist(), "p1": xy[-1].tolist(),
                            "r": float(np.median(np.linalg.norm(xy - c, axis=1)))})
        if els:
            out.append({"type": "loop", "elements": els})
    return out


def one(part):
    from photo2fcstd import analysis, bench, constraints as K, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        loops = (doc.get("outline") or {}).get("loops")
        if not loops:
            return None
        ref = ideal_loops(IDEAL[part])
        if not ref:
            return None
        lab = K.labels_for(loops, ref)
        if lab["n"] < 3:
            return None
        els = lab["elements"]
        pairs = []
        for i in range(len(els)):
            for j in range(i + 1, len(els)):
                for kind in ("equal_length", "parallel", "perpendicular", "equal_radius"):
                    v = lab["pair_label"](kind, i, j)
                    if v is not None:
                        pairs.append((kind, i, j, int(v)))
        return {"part": part, "n": lab["n"], "matched": lab["matched"].tolist(),
                "dist": lab["dist"].tolist(), "angle": lab["angle"],
                "elements": els, "pairs": pairs}
    except Exception:
        return None


def main(limit=700):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    json.dump(rows, open(os.path.join(ROOT, "data", "constraint_rows.json"), "w"))
    d = np.concatenate([np.array(r["dist"]) for r in rows])
    m = np.concatenate([np.array(r["matched"]) for r in rows])
    npairs = sum(len(r["pairs"]) for r in rows)
    print("%d parts, %d elements, %d labelled pairs\n" % (len(rows), len(d), npairs))
    print("  label quality - distance from a traced element to its ideal partner,")
    print("  in units of the sketch diagonal:")
    for q in (50, 75, 90):
        print("    p%-3d %.4f" % (q, np.percentile(d, q)))
    print("    matched within 0.05: %.0f%%" % (100 * m.mean()))
    ang = [a for r in rows for a in r["angle"]]
    known = [a for a in ang if a is not None]
    print("\n  angle labels: %d of %d elements matched" % (len(known), len(ang)))
    from collections import Counter
    print("    %s" % dict(Counter(known)))
    kinds = Counter(k for r in rows for k, *_ in r["pairs"])
    pos = Counter(k for r in rows for k, _, _, v in r["pairs"] if v)
    print("\n  pair labels:")
    for k in kinds:
        print("    %-14s %6d labelled, %5.1f%% positive" % (k, kinds[k], 100 * pos[k] / kinds[k]))


if __name__ == "__main__":
    main()
