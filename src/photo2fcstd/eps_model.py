"""Choose the corner tolerance per part instead of using one for everything.

`approxPolyDP` takes a single tolerance, so it cannot be loose enough for a large outline and tight
enough for a small notch on it at the same time: part 00911's four corner notches survive at 0.010
and are gone at the shipped 0.015. Choosing per part is worth +0.024 [+0.017, +0.031] of sketch IoU
against an oracle, and the shipped value is the best one on only 15% of parts.

Posed as selection among five candidates scored end to end, which is the shape of the two learned
components here that worked, rather than as a geometric label needing correspondence to a reference,
which is the shape of the five that did not.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_EPS_MODEL_PATH", os.path.join(ROOT, "data", "eps_model.joblib"))
CANDIDATES = (0.006, 0.010, 0.015, 0.022, 0.030)
DEFAULT = 0.015
ENABLED = os.environ.get("P2F_EPS_MODEL", "0") == "1"
_CACHE = {}


def per_candidate(view, contour, counts):
    """One row per tolerance, describing that tolerance rather than the part alone.

    The 5-way classifier has to learn each tolerance's behaviour from scratch; scoring candidates
    independently shares strength across them and gives five rows per part instead of one. That is
    the framing both learned components here that worked already use.
    """
    base = features_for(view, contour, counts)
    n = np.array(counts, float)
    rows = []
    for i, e in enumerate(CANDIDATES):
        c = float(counts[i])
        rows.append(base + [e, c, c / max(n.max(), 1.0), c - float(np.median(n)),
                            float(i), abs(c - float(np.median(n))),
                            c / max(float(counts[max(i - 1, 0)]), 1.0),
                            c / max(float(counts[min(i + 1, len(counts) - 1)]), 1.0)])
    return np.asarray(rows, float)


def features_for(view, contour, counts):
    """The part's own shape, plus how much the element count moves as the tolerance changes.

    A part whose count collapses from 16 to 4 across the range is one where the choice matters;
    a circle that gives the same answer everywhere is one where it does not.
    """
    s = view["shape"]
    c = np.asarray(contour, float)
    box = c.max(axis=0) - c.min(axis=0)
    n = np.array(counts, float)
    return [float(s["rectangularity"]), float(s["solidity"]), float(view["elongation"]),
            float(s["hole_frac"]), float(s["ellipse_rms"]),
            float(s["stroke_px"]) / max(float(view["length_px"]), 1.0),
            float(len(s.get("holes", []))), float(len(c)),
            float(min(box) / max(max(box), 1e-9)), float(view["length_px"]),
            float(n.max()), float(n.min()), float(n.max() - n.min()),
            float(n.max() / max(n.min(), 1.0)), float(np.median(n)), float(n.std()),
            ] + list(n)


def load():
    if "m" in _CACHE:
        return _CACHE["m"]
    from photo2fcstd import fallback
    m = None
    if not os.path.exists(MODEL_PATH):
        fallback.note("eps_model", "no model at %s" % MODEL_PATH)
    else:
        try:
            import joblib
            m = joblib.load(MODEL_PATH)
        except Exception as exc:
            fallback.note("eps_model", "could not load %s: %s" % (MODEL_PATH, exc))
    _CACHE["m"] = m
    return m


def element_counts(contour, length_px):
    from photo2fcstd import trace
    out = []
    keep = trace.RUN_EPS
    try:
        for e in CANDIDATES:
            trace.RUN_EPS = e
            try:
                out.append(len(trace.corner_runs(np.asarray(contour, float))))
            except Exception:
                out.append(0)
    finally:
        trace.RUN_EPS = keep
    return out


def choose(view, contour):
    """The tolerance to trace this part with, or None when no model is installed."""
    if not ENABLED:
        return None
    m = load()
    if m is None:
        return None
    try:
        counts = element_counts(contour, view["length_px"])
        x = per_candidate(view, contour, counts)
        return float(CANDIDATES[int(np.argmax(m.predict_proba(x)[:, 1]))])
    except Exception as exc:
        from photo2fcstd import fallback
        fallback.note("eps_model", "scoring failed: %s" % exc)
        return None
