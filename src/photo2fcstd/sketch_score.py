import json
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

CURVES = ("arc", "circle", "ellipse", "bsplinecurve")


def chain_edges(segments, tol_frac=0.02):
    """Walk edges end to end. A STEP wire lists its edges in topological order and each
    may run either way, so concatenating them as stored can scramble the ring."""
    segs = [np.asarray(s, float) for s in segments if len(s) >= 2]
    if len(segs) < 2:
        return np.vstack(segs) if segs else np.zeros((0, 2))
    chain = [segs.pop(0)]
    while segs:
        end = chain[-1][-1]
        best = min(((min(np.linalg.norm(end - s[0]), np.linalg.norm(end - s[-1])), i,
                     np.linalg.norm(end - s[-1]) < np.linalg.norm(end - s[0]))
                    for i, s in enumerate(segs)), key=lambda t: t[0])
        _, i, flip = best
        s = segs.pop(i)
        chain.append(s[::-1] if flip else s)
    return np.vstack(chain)


def ideal_rings(record):
    return [chain_edges([e["xy"] for e in loop if e.get("xy")])
            for loop in record.get("loops", []) if loop]


def spec_rings(spec):
    if spec.get("revolve"):
        v = spec["revolve"]
        prof = np.array(v["profile"], float)
        R = float(np.max(np.abs(prof[:, 0]))) or 1.0
        bore = float(np.min(np.abs(prof[:, 0])))
        rings = [circle_ring(0.0, 0.0, R)]
        if bore > 0.02 * R:
            rings.append(circle_ring(0.0, 0.0, bore))
        return rings + [circle_ring(h["cx"], h["cy"], h["r"]) for h in v["holes"] if h["type"] == "circle"]
    if spec.get("outline"):
        return [loop_ring(l) for l in spec["outline"]["loops"]]
    return []


def circle_ring(cx, cy, r, n=96):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def arc_ring(e, step=math.radians(4)):
    a0 = math.atan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"])
    a1 = math.atan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"])
    if e["ccw"]:
        while a1 <= a0:
            a1 += 2 * math.pi
    else:
        while a0 <= a1:
            a0 += 2 * math.pi
    ang = np.linspace(a0, a1, max(int(abs(a1 - a0) / step), 2))
    return np.column_stack([e["cx"] + e["r"] * np.cos(ang), e["cy"] + e["r"] * np.sin(ang)])


def loop_ring(loop):
    if loop["type"] == "circle":
        return circle_ring(loop["cx"], loop["cy"], loop["r"])
    return np.vstack([np.array([e["p0"], e["p1"]], float) if e["type"] == "line" else arc_ring(e)
                      for e in loop["elements"]])


def polygon_of(rings):
    rings = [r for r in rings if len(r) >= 4]
    if not rings:
        return None
    outer = max(rings, key=lambda r: Polygon(r).buffer(0).area)
    holes = [r for r in rings if r is not outer]
    p = Polygon(outer, holes)
    return p if p.is_valid else p.buffer(0)


def normalise(poly):
    x0, y0, x1, y1 = poly.bounds
    lo, hi = np.array([x0, y0]), np.array([x1, y1])
    k = 1.0 / max(np.max(hi - lo), 1e-9)
    import shapely.affinity as aff
    c = (lo + hi) / 2
    return aff.scale(aff.translate(poly, -c[0], -c[1]), k, k, origin=(0, 0))


DIHEDRAL = [(1, 1, False), (1, -1, False), (-1, 1, False), (-1, -1, False),
            (1, 1, True), (1, -1, True), (-1, 1, True), (-1, -1, True)]


def posed(poly, sx, sy, swap):
    import shapely.affinity as aff
    p = aff.scale(poly, sx, sy, origin=(0, 0))
    return aff.affine_transform(p, [0, 1, 1, 0, 0, 0]) if swap else p


def region_iou(a, b):
    if a is None or b is None:
        return 0.0
    na, nb = normalise(a), normalise(b)
    best = 0.0
    for sx, sy, swap in DIHEDRAL:
        q = posed(nb, sx, sy, swap)
        inter = na.intersection(q).area
        union = na.union(q).area
        best = max(best, inter / union if union else 0.0)
    return float(best)


def best_aligned(a, b):
    na, nb = normalise(a), normalise(b)
    best, bq = -1.0, nb
    for pose in DIHEDRAL:
        q = posed(nb, *pose)
        u = na.union(q).area
        v = na.intersection(q).area / u if u else 0.0
        if v > best:
            best, bq = v, q
    return na, bq, best


def boundary_deviation(a, b, n=240):
    import numpy as _np
    from shapely.geometry import Point
    def sample(poly):
        ring = poly.boundary
        return [ring.interpolate(t, normalized=True) for t in _np.linspace(0, 1, n, endpoint=False)]
    d1 = [b.boundary.distance(p) for p in sample(a)]
    d2 = [a.boundary.distance(p) for p in sample(b)]
    d = _np.array([x for x in d1 + d2 if _np.isfinite(x)])
    if not len(d):
        return 0.0, 0.0
    return float(d.mean()), float(_np.percentile(d, 95))


def compare(a, b):
    if a is None or b is None:
        return {"iou": 0.0, "missing": 1.0, "extra": 0.0, "dev_mean": 1.0, "dev_p95": 1.0}
    na, nb, iou = best_aligned(a, b)
    ref = nb.area or 1e-9
    dev_mean, dev_p95 = boundary_deviation(na, nb)
    return {"iou": iou,
            "missing": float(nb.difference(na).area / ref),
            "extra": float(na.difference(nb).area / ref),
            "dev_mean": dev_mean, "dev_p95": dev_p95}


def counts_of_spec(spec):
    c = {}
    if spec.get("revolve"):
        return {"circle": len(spec["revolve"]["holes"]) + 1}
    for loop in (spec.get("outline") or {}).get("loops", []):
        if loop["type"] == "circle":
            c["circle"] = c.get("circle", 0) + 1
        else:
            for e in loop["elements"]:
                c[e["type"]] = c.get(e["type"], 0) + 1
    return c


def curve_fraction(counts):
    tot = sum(counts.values())
    return sum(v for k, v in counts.items() if k in CURVES) / tot if tot else 0.0


def face_aspect(record):
    p = np.array([q for lp in record.get("loops", []) for e in lp for q in e.get("xy", [])], float)
    if len(p) < 3:
        return 0.0
    b = p.max(axis=0) - p.min(axis=0)
    return float(min(b) / max(b)) if max(b) > 0 else 0.0


def trustworthy(record):
    if not record.get("prism") or "loops" not in record:
        return False
    if face_aspect(record) < 0.15:
        return False
    a, d, vol = record.get("face_area"), record.get("depth"), record.get("volume")
    return not (a and d and vol and a * d / vol > 3)


def score_one(spec, record):
    rings = spec_rings(spec)
    mine = polygon_of(rings)
    ideal = polygon_of(ideal_rings(record))
    mc, ic = counts_of_spec(spec), record.get("counts", {})
    cmp = compare(mine, ideal)
    return {"has_sketch": bool(rings), "region_iou": cmp["iou"],
            "missing": cmp["missing"], "extra": cmp["extra"],
            "dev_mean": cmp["dev_mean"], "dev_p95": cmp["dev_p95"],
            "loops_mine": len(rings), "loops_ideal": record.get("n_loops", 0),
            "curve_frac_mine": curve_fraction(mc), "curve_frac_ideal": curve_fraction(ic),
            "counts_mine": mc, "counts_ideal": ic, "prism": bool(record.get("prism")),
            "trustworthy": trustworthy(record)}


def main(argv):
    run_dir, ideal_path = argv[0], argv[1] if len(argv) > 1 else "data/printcad_ideal_sketches_all.json"
    ideal = json.load(open(ideal_path))
    import glob
    rows = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*.spec.json"))):
        k = os.path.basename(p).split(".")[0]
        if k not in ideal or "loops" not in ideal[k]:
            continue
        try:
            rows[k] = score_one(json.load(open(p)), ideal[k])
        except Exception as exc:
            rows[k] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:80])}
    ok = [r for r in rows.values() if "error" not in r]
    good = [r for r in ok if r["trustworthy"]]
    drawn = [r for r in ok if r["has_sketch"]]
    print("sketch score over %d parts (%d errors)" % (len(ok), len(rows) - len(ok)))
    print("  emit a sketch          %d (%.0f%%)" % (len(drawn), 100 * len(drawn) / max(len(ok), 1)))
    print("  region IoU  all parts  %.3f" % np.mean([r["region_iou"] for r in ok]))
    print("  region IoU  trustworthy ground truth (n=%d)  %.3f"
          % (len(good), np.mean([r["region_iou"] for r in good]) if good else 0.0))
    print("  region IoU  when drawn %.3f" % (np.mean([r["region_iou"] for r in drawn]) if drawn else 0))
    print("  loop count exact       %.0f%%" % (100 * np.mean([r["loops_mine"] == r["loops_ideal"] for r in drawn]) if drawn else 0))
    print("  curve fraction ours %.2f  ideal %.2f" % (np.mean([r["curve_frac_mine"] for r in drawn]) if drawn else 0,
                                                      np.mean([r["curve_frac_ideal"] for r in drawn]) if drawn else 0))
    return rows


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
