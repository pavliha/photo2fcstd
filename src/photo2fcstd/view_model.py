"""Pick which of the photos to draw from.

Estimating viewpoint tilt from a single silhouette is ambiguous - every hand-written criterion
for it scores below leaving the photo alone. Choosing among the photos already taken is a
different problem: the answer is one of three, the ceiling is +0.088 of sketch IoU, and the same
per-candidate scoring that fixed the carve axis applies.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_VIEW_MODEL_PATH", os.path.join(ROOT, "data", "view_model.joblib"))
FIELDS = ("rect", "sol", "elong", "ellipse_rms", "hole_frac", "stroke", "nholes", "area")
_CACHE = {}


def stats_of_view(view):
    """Read the pre-regularisation statistics when the view carries them."""
    s = view["shape"]
    picked = view.get("select") or {}
    get = lambda key, default: picked.get(key, default)
    return {"rect": get("rectangularity", s["rectangularity"]), "sol": get("solidity", s["solidity"]),
            "elong": get("elongation", view["elongation"]), "ellipse_rms": s["ellipse_rms"],
            "hole_frac": get("hole_frac", s["hole_frac"]), "stroke": get("stroke_px", s["stroke_px"]),
            "nholes": get("nholes", len(s["holes"])), "area": float(get("length_px", view["length_px"]))}


def features(per_view):
    """Each view described by its own statistics and by how it compares with its siblings."""
    rows = [[float(v[f]) for f in FIELDS] for v in per_view]
    a = np.asarray(rows, float)
    med = np.median(a, axis=0)
    rng = np.maximum(a.max(axis=0) - a.min(axis=0), 1e-9)
    out = []
    for i in range(len(a)):
        rel = (a[i] - med) / rng
        rank = [float((a[:, j] <= a[i, j]).sum()) / len(a) for j in range(a.shape[1])]
        out.append(np.concatenate([a[i], rel, rank, [len(a)]]))
    return np.asarray(out, float)


def load():
    if "m" in _CACHE:
        return _CACHE["m"]
    from photo2fcstd import fallback
    m = None
    if not os.path.exists(MODEL_PATH):
        fallback.note("view_model", "no model at %s" % MODEL_PATH)
    else:
        try:
            import joblib
            m = joblib.load(MODEL_PATH)
        except Exception as exc:
            fallback.note("view_model", "could not load %s: %s" % (MODEL_PATH, exc))
    _CACHE["m"] = m
    return m


def choose(views):
    """Index of the view to draw from, or None when no model is installed."""
    m = load()
    if m is None or len(views) < 2:
        return None
    try:
        x = features([stats_of_view(v) for v in views])
        return int(np.argmax(m.predict_proba(x)[:, 1]))
    except Exception as exc:
        from photo2fcstd import fallback
        fallback.note("view_model", "scoring failed: %s" % exc)
        return None
