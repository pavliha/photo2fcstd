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


def primitive_f1(mine, ideal):
    """Precision and recall over primitive counts, so over-drawing and under-drawing cost alike.

    Region IoU rewards an outline that agrees and cannot see a missing feature; exact-count
    rewards precision and gives nothing for getting close. This scores both directions of the
    same error with one number, matching primitives type by type.
    """
    drawn = sum(mine.values()) if mine else 0
    wanted = sum(ideal.values()) if ideal else 0
    if not drawn or not wanted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "drawn": drawn, "wanted": wanted}
    matched = sum(min(mine.get(k, 0), ideal.get(k, 0)) for k in set(mine) | set(ideal))
    precision = matched / drawn
    recall = matched / wanted
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "drawn": drawn, "wanted": wanted}


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
    triv = trivial_score(record)
    return {"has_sketch": bool(rings), "region_iou": cmp["iou"],
            "trivial": triv, "difficulty": 1.0 - triv, "discriminating": (1.0 - triv) >= 0.15,
            "missing": cmp["missing"], "extra": cmp["extra"],
            "dev_mean": cmp["dev_mean"], "dev_p95": cmp["dev_p95"],
            "loops_mine": len(rings), "loops_ideal": record.get("n_loops", 0),
            "primitive_f1": primitive_f1(mc, ic),
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


def _mask(poly, n=192):
    from shapely import contains_xy
    x0, y0, x1, y1 = poly.bounds
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys)
    return contains_xy(poly, gx.ravel(), gy.ravel()).reshape(n, n), (xs, ys)


def equivalent_ellipse(poly):
    from shapely.geometry import Point
    import shapely.affinity as aff
    m, (xs, ys) = _mask(poly)
    if m.sum() < 4:
        return None
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx[m], gy[m]
    cov = np.cov(np.stack([px, py]))
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    a, b = 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0])
    k = np.sqrt(poly.area / max(np.pi * a * b, 1e-12))
    e = aff.scale(Point(0, 0).buffer(1, quad_segs=64), k * a, k * b, origin=(0, 0))
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    return aff.translate(aff.rotate(e, ang, origin=(0, 0)), px.mean(), py.mean())


def trivial_score(record):
    """What a pipeline that always draws one circle would score on this part.

    Region IoU is scale-invariant, so any circle matches any circle at about 0.99. This is
    the score to beat before a number means anything, and it is a real answer a pipeline
    could produce without looking at the photo - unlike a shape fitted to the reference,
    which is not a baseline but a second oracle.
    """
    from shapely.geometry import Point
    ideal = polygon_of(ideal_rings(record))
    if ideal is None or ideal.is_empty:
        return 0.0
    return region_iou(ideal, Point(0, 0).buffer(1, quad_segs=64))


def difficulty(record):
    return 1.0 - trivial_score(record)


def discriminating(record, floor=0.15):
    return difficulty(record) >= floor


def skill(iou, baseline):
    """Fraction of the achievable margin above a trivial answer that was actually won."""
    return float((iou - baseline) / (1.0 - baseline)) if baseline < 1.0 else 0.0


def structure_score(result):
    """How much of the real sketch's structure a drawing reproduces.

    Region IoU decides every A/B here and correlates 0.22 with structural agreement, 0.33 with
    reproducing the exact primitives: it cannot see primitive type, because an arc and the chords
    approximating it cover the same area, and it cannot see a small hole, because a hole carries
    almost no area. Loops catch the missed hole, curves the arc drawn as chords, elements the
    fragmentation. Unweighted - weighting needs someone to say which drawing they would rather edit.
    """
    mine, ideal = result.get("counts_mine") or {}, result.get("counts_ideal") or {}
    total_m, total_i = sum(mine.values()), sum(ideal.values())
    cm = sum(v for k, v in mine.items() if k in CURVES)
    ci = sum(v for k, v in ideal.items() if k in CURVES)
    loops = min(result.get("loops_mine", 0), result.get("loops_ideal", 0)) / max(result.get("loops_ideal", 0), 1)
    curves = 1.0 - abs(cm - ci) / max(ci, cm, 1)
    elements = max(1.0 - abs(total_m - total_i) / max(total_i, 1), 0.0)
    return {"loops": float(loops), "curves": float(curves), "elements": float(elements),
            "structure": float((loops + curves + elements) / 3)}


def verdict(result):
    """What a change did, in the terms a change should be judged on.

    Region IoU stays because it is comparable with everything already published here, but a drawing
    that gains area and loses primitives has not improved.
    """
    s = structure_score(result)
    return {"region_iou": result.get("region_iou", 0.0),
            "exact": result.get("counts_mine") == result.get("counts_ideal"),
            **s, "trivial": result.get("trivial", 0.0),
            "discriminating": (1.0 - result.get("trivial", 0.0)) >= 0.15}
