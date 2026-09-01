"""Which constraints hold between the elements of a sketch.

Applying a CAD prior uniformly loses: enforcing mirror symmetry on every near-symmetric outline
costs 0.021 of sketch IoU and 4.5 points of exact primitives, because the prior is true of the part
and false of the tracing - when one side is seen well and the other badly, averaging drags the good
side down. The alternative is to decide per edge and per pair, which needs labels, and the STEP
sketches provide them exactly.
"""
import numpy as np

HORIZONTAL, VERTICAL, DIAGONAL = "horizontal", "vertical", "diagonal"
ANGLE_TOL_DEG = 2.0
EQUAL_TOL = 0.03


def edge_vector(e):
    if e.get("type") == "line":
        return np.array(e["p1"], float) - np.array(e["p0"], float)
    return None


def edge_length(e):
    d = edge_vector(e)
    if d is not None:
        return float(np.hypot(*d))
    return float(abs(e.get("r", 0.0)) * np.radians(abs(e.get("span_deg", 90.0))))


def edge_angle(e):
    d = edge_vector(e)
    return None if d is None else float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def angle_class(e, tol=ANGLE_TOL_DEG, step=15.0):
    """Which canonical direction a straight edge sits on, or None when it sits on none."""
    a = edge_angle(e)
    if a is None:
        return None
    if min(a, 180 - a) <= tol:
        return HORIZONTAL
    if abs(a - 90) <= tol:
        return VERTICAL
    target = round(a / step) * step
    return DIAGONAL if abs(a - target) <= tol and abs(target % 90) > 1e-6 else None


def loop_elements(loop):
    if loop.get("type") == "circle":
        return [{"type": "circle", "r": float(loop["r"]), "cx": loop["cx"], "cy": loop["cy"]}]
    return list(loop.get("elements", []))


def equal_pairs(els, tol=EQUAL_TOL):
    """Index pairs whose lengths match to within tol of the larger - the same slot cut twice."""
    n = len(els)
    lens = [edge_length(e) for e in els]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lens[i], lens[j]
            if max(a, b) > 0 and abs(a - b) / max(a, b) <= tol:
                out.append((i, j))
    return out


def equal_radius_pairs(els, tol=EQUAL_TOL):
    out = []
    idx = [i for i, e in enumerate(els) if e.get("type") in ("arc", "circle") and e.get("r")]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            ra, rb = float(els[i]["r"]), float(els[j]["r"])
            if max(ra, rb) > 0 and abs(ra - rb) / max(ra, rb) <= tol:
                out.append((i, j))
    return out


def parallel_pairs(els, tol=ANGLE_TOL_DEG):
    out = []
    ang = [edge_angle(e) for e in els]
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            if ang[i] is None or ang[j] is None:
                continue
            d = abs(((ang[i] - ang[j] + 90) % 180) - 90)
            if d <= tol:
                out.append((i, j))
    return out


def perpendicular_pairs(els, tol=ANGLE_TOL_DEG):
    out = []
    ang = [edge_angle(e) for e in els]
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            if ang[i] is None or ang[j] is None:
                continue
            d = abs(((ang[i] - ang[j] - 90 + 90) % 180) - 90)
            if d <= tol:
                out.append((i, j))
    return out


def of_sketch(loops):
    """Every constraint that holds in one sketch, as element indices within the flattened list."""
    els, offset, spans = [], 0, []
    for lp in loops:
        e = loop_elements(lp)
        spans.append((offset, offset + len(e)))
        els += e
        offset += len(e)
    return {"elements": els, "loops": spans,
            "angle": [angle_class(e) for e in els],
            "equal_length": equal_pairs(els),
            "equal_radius": equal_radius_pairs(els),
            "parallel": parallel_pairs(els),
            "perpendicular": perpendicular_pairs(els)}


def element_midpoints(els):
    out = []
    for e in els:
        if e.get("type") == "line":
            out.append((np.array(e["p0"], float) + np.array(e["p1"], float)) / 2)
        elif e.get("type") == "circle":
            out.append(np.array([e["cx"], e["cy"]], float))
        else:
            p0, p1 = np.array(e.get("p0", [0, 0]), float), np.array(e.get("p1", [0, 0]), float)
            out.append((p0 + p1) / 2)
    return np.asarray(out, float) if out else np.zeros((0, 2))


def align(mine, ideal):
    """Match each traced element to an ideal one, under the pose that best overlaps the two.

    Labels are only worth training on if this correspondence is right, so it reuses the dihedral
    search `sketch_score` already uses to compare sketches, then matches on midpoint distance in
    the shared normalised frame. Returns (partner index, distance) per traced element, with the
    distance in units of the sketch diagonal so it can be thresholded.
    """
    import shapely.affinity as aff
    from shapely.geometry import Polygon
    from photo2fcstd.sketch_score import DIHEDRAL, normalise, posed

    a, b = element_midpoints(mine), element_midpoints(ideal)
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0)

    def frame(pts):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        k = 1.0 / max(np.max(hi - lo), 1e-9)
        return (pts - (lo + hi) / 2) * k

    fa, fb = frame(a), frame(b)
    best = None
    for sx, sy, swap in DIHEDRAL:
        q = fa * np.array([sx, sy])
        if swap:
            q = q[:, ::-1]
        d = np.linalg.norm(q[:, None, :] - fb[None, :, :], axis=2)
        cost = float(d.min(axis=1).mean())
        if best is None or cost < best[0]:
            best = (cost, d)
    d = best[1]
    return d.argmin(axis=1), d.min(axis=1)


def labels_for(mine_loops, ideal_loops, max_dist=0.05):
    """Per traced element, the constraints its ideal counterpart satisfies. None when unmatched."""
    mine = of_sketch(mine_loops)
    ref = of_sketch(ideal_loops)
    partner, dist = align(mine["elements"], ref["elements"])
    ok = dist <= max_dist
    ang = [ref["angle"][partner[i]] if ok[i] else None for i in range(len(partner))]
    ref_pairs = {k: set(map(tuple, ref[k])) for k in ("equal_length", "equal_radius", "parallel", "perpendicular")}

    def pair_label(kind, i, j):
        if not (ok[i] and ok[j]):
            return None
        p, q = int(partner[i]), int(partner[j])
        if p == q:
            return None
        return (min(p, q), max(p, q)) in ref_pairs[kind]

    return {"angle": ang, "matched": ok, "dist": dist, "pair_label": pair_label,
            "n": len(mine["elements"]), "elements": mine["elements"]}
