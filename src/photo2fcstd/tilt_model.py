"""How far off square a photograph was taken, without a ChArUco board in the frame.

Shooting square to the face is worth about +0.13 of sketch IoU and needs no code at all, which
makes it the cheapest improvement available here - but only if the photographer is told. Until now
that needed the printed target, because tilt came from a solved board pose.

The tilt cannot be corrected once shot: at 15 degrees the projective distortion a homography
removes costs -0.009 [-0.026, +0.008] while the side walls coming into view cost -0.084
[-0.114, -0.058], and rectifying by the *true* normal scores -0.013. So this reports, it does not
repair, and `preflight` uses it to ask for a reshoot.
"""
import os

import numpy as np

from photo2fcstd import fallback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.environ.get("P2F_TILT_MODEL_PATH", os.path.join(ROOT, "data", "tilt_model.joblib"))
ENABLED = os.environ.get("P2F_TILT_MODEL", "1") != "0"
WARN_DEG = float(os.environ.get("P2F_TILT_WARN_DEG", "8"))
_CACHE = {}


def model():
    if not ENABLED:
        return None
    if "m" not in _CACHE:
        if not os.path.exists(MODEL_PATH):
            fallback.note("tilt_model", "no artefact at %s; run tools/tilt_train.py" % MODEL_PATH)
            _CACHE["m"] = None
        else:
            import joblib
            _CACHE["m"] = joblib.load(MODEL_PATH)
    return _CACHE["m"]


def is_familiar(vector, m):
    """Refuse a photograph unlike the ones the head was fitted on.

    The T-LESS head reaches 4.8 degrees held out by object and 34 degrees on PrintCAD, and its gate
    refuses every PrintCAD image - so the gate is the difference between an honest abstention and
    the depth head's confidently wrong number.
    """
    return float(1.0 - vector @ m["centre"]) <= m["gate"]


def normal_of(path):
    """The part's face normal in camera coordinates, or None if the head cannot speak."""
    from photo2fcstd import embed
    m = model()
    if m is None:
        return None
    vectors = embed.vectors_for([path])
    if vectors is None:
        return None
    v = vectors[0]
    if not is_familiar(v, m):
        fallback.note("tilt_model", "photograph is outside the head's training distribution")
        return None
    x = ((v - m["mu"]) / m["sd"]).reshape(1, -1)
    n = np.array([float(h.predict(x)[0]) for h in m["heads"]])
    norm = float(np.linalg.norm(n))
    return None if norm < 1e-9 else n / norm


def tilt_of(path):
    """Degrees off square, or None."""
    n = normal_of(path)
    return None if n is None else float(np.degrees(np.arccos(min(abs(n[2]), 1.0))))


def square_check(paths):
    """Problems and notes for `preflight`, with no board required."""
    m = model()
    tilts = [(p, tilt_of(p)) for p in paths]
    known = [(p, t) for p, t in tilts if t is not None]
    if not known:
        return [], ["no board and no usable tilt estimate, so squareness was not checked"]
    err = m["held_out_mae"] if m else 0.0
    off = [(p, t) for p, t in known if t > WARN_DEG]
    problems, notes = [], []
    if off:
        problems.append("%d of %d photographs are more than %.0f degrees off square (worst %.0f). "
                        "Tilt cannot be corrected afterwards - eight degrees already costs about "
                        "0.11 of sketch IoU - so reshoot those square to the face"
                        % (len(off), len(known), WARN_DEG, max(t for _, t in off)))
    else:
        notes.append("all %d photographs are within %.0f degrees of square (worst %.0f)"
                     % (len(known), WARN_DEG, max(t for _, t in known)))
    if len(known) < len(paths):
        notes.append("%d photographs were outside the head's training distribution and not checked"
                     % (len(paths) - len(known)))
    notes.append("tilt estimates carry about %.1f degrees of error" % err)
    return problems, notes
