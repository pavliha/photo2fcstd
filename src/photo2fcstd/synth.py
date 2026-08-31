import json
import os

import numpy as np

CLASSES = ("straight", "curved")
LABEL = {"line": 0, "arc": 1, "circle": 1, "bsplinecurve": 1, "ellipse": 1}
CANVAS = 768


def loop_edges(loop):
    return [(e["type"], np.asarray(e["xy"], float)) for e in loop if e.get("xy") and len(e["xy"]) >= 2]


def homography(rng, tilt_deg=38.0):
    a, b = np.radians(rng.uniform(-tilt_deg, tilt_deg, 2))
    f = rng.uniform(2.5, 9.0)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    R = np.array([[cb, sb * sa, sb * ca], [0, ca, -sa], [-sb, cb * sa, cb * ca]])
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    H = K @ R
    th = rng.uniform(0, 2 * np.pi)
    S = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    return H @ S


def apply_h(H, pts):
    q = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def fit_canvas(loops, n=CANVAS, margin=0.08):
    allp = np.vstack([p for _, p in sum(loops, [])])
    lo, hi = allp.min(0), allp.max(0)
    k = (1 - 2 * margin) * n / max(np.max(hi - lo), 1e-9)
    off = n / 2 - (lo + hi) / 2 * k
    return [[(t, p * k + off) for t, p in loop] for loop in loops]


def rasterise(loops, n=CANVAS):
    import cv2
    img = np.zeros((n, n), np.uint8)
    rings = [np.round(np.vstack([p for _, p in loop])).astype(np.int32) for loop in loops]
    if rings:
        cv2.fillPoly(img, rings[:1], 1)
        for r in rings[1:]:
            cv2.fillPoly(img, [r], 0)
    return img > 0


def segment_distance(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


AMBIGUOUS_PX = 2.5


def label_contour(contour, loops):
    per = [np.full(len(contour), np.inf), np.full(len(contour), np.inf)]
    for loop in loops:
        for t, p in loop:
            cls = LABEL.get(t, 0)
            for i in range(len(p) - 1):
                d = segment_distance(contour, p[i], p[i + 1])
                per[cls] = np.minimum(per[cls], d)
    d0, d1 = per
    lab = (d1 < d0).astype(int)
    best = np.minimum(d0, d1)
    ambiguous = np.abs(d0 - d1) < AMBIGUOUS_PX
    return lab, best, ambiguous


def sample(record, rng, n=CANVAS):
    from photo2fcstd.trace import outline
    loops = [loop_edges(lp) for lp in record["loops"]]
    loops = [lp for lp in loops if lp]
    if not loops:
        return None
    H = homography(rng)
    loops = [[(t, apply_h(H, p)) for t, p in lp] for lp in loops]
    loops = fit_canvas(loops, n)
    mask = rasterise(loops, n)
    if mask.sum() < 500:
        return None
    poly, shape = outline(mask)
    contour = np.asarray(shape["raw"], float)
    if len(contour) < 12:
        return None
    y, dist, amb = label_contour(contour, loops)
    heat, corners = corner_heatmap(contour, loops)
    return {"contour": contour, "label": y, "dist": dist, "amb": amb, "n": n,
            "heat": heat, "corners": corners}


def build(ideal_path, out_path, per_part=4, limit=None, seed=0):
    from photo2fcstd.sketch_score import trustworthy
    ideal = json.load(open(ideal_path))
    parts = [k for k in sorted(ideal) if trustworthy(ideal[k])]
    parts = parts[:limit] if limit else parts
    rng = np.random.default_rng(seed)
    out = []
    for i, k in enumerate(parts):
        for _ in range(per_part):
            try:
                s = sample(ideal[k], rng)
            except Exception:
                s = None
            if s and (s["dist"] < 8.0).mean() > 0.85:
                out.append({"part": k, "contour": s["contour"].round(2).tolist(),
                            "label": s["label"].astype(int).tolist(),
                            "amb": s["amb"].astype(np.uint8).tolist(),
                            "heat": s["heat"].round(4).tolist(),
                            "corners": s["corners"].round(2).tolist()})
    np.save(out_path, np.array(out, dtype=object), allow_pickle=True)
    return out


def demo():
    ideal = json.load(open("data/printcad_ideal_sketches_all.json"))
    from photo2fcstd.sketch_score import trustworthy
    k = next(x for x in sorted(ideal) if trustworthy(ideal[x]) and ideal[x]["n_loops"] > 1)
    s = sample(ideal[k], np.random.default_rng(0))
    assert s is not None, "no sample"
    assert len(s["contour"]) == len(s["label"])
    assert (s["dist"] < 8.0).mean() > 0.8, s["dist"].mean()
    print("synth self-check ok: part %s, %d points, labels %s, ambiguous %.1f%%, median dist %.2f px"
          % (k, len(s["contour"]), np.bincount(s["label"], minlength=2).tolist(),
             100 * s["amb"].mean(), np.median(s["dist"])))


if __name__ == "__main__":
    demo()


CORNER_SIGMA_PX = 6.0


def loop_corners(loops):
    """Where one primitive hands over to the next - the points a line endpoint has to land on."""
    out = []
    for loop in loops:
        pts = [p for _, p in loop if len(p) >= 2]
        if not pts:
            continue
        if len(pts) == 1:
            continue
        out += [p[0] for p in pts]
    return np.asarray(out, float) if out else np.zeros((0, 2))


def corner_heatmap(contour, loops, sigma=CORNER_SIGMA_PX):
    """A soft target on the contour, peaked where a corner is.

    Labelling each point straight or curved was tried and lost end to end: a per-point
    boundary is fuzzy by several points, and a corner off by a few points moves a line
    endpoint visibly. A peak can be decoded to a position between two points, which is what
    `approxPolyDP` gives and what the fitter actually consumes.
    """
    c = np.asarray(contour, float)
    corners = loop_corners(loops)
    if len(corners) == 0 or len(c) < 3:
        return np.zeros(len(c)), corners
    d = np.linalg.norm(c[:, None, :] - corners[None, :, :], axis=2).min(axis=1)
    return np.exp(-0.5 * (d / sigma) ** 2), corners
