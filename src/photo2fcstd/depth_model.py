import os

import numpy as np

FIELDS = ("elongation", "rectangularity", "solidity", "hole_frac", "min_over_max",
          "ellipse_rms", "stroke_px", "length_px", "stations")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_DEPTH_MODEL", os.path.join(ROOT, "data", "depth_model.joblib"))
PIXEL_PATH = os.environ.get("P2F_DEPTH_PIXEL_MODEL", os.path.join(ROOT, "data", "depth_pixel_model.joblib"))
HYBRID_PATH = os.path.join(ROOT, "data", "depth_hybrid_model.joblib")
USE_PIXELS = os.environ.get("P2F_DEPTH_PIXELS", "1") == "1"
ALLOW_BACKBONE = os.environ.get("P2F_EMBED_BACKBONE", "1") == "1"
_CACHE = {}


def features(events):
    v = sorted(events, key=lambda x: -x["elongation"])[:3]
    row = []
    for x in v:
        row += [float(x[f]) for f in FIELDS]
        row += [float(x["ellipse_aspect"] or 0.0), float(x["round"]), float(x["roundish"]),
                float(x["stroke_px"]) / max(float(x["length_px"]), 1.0), float(len(x["symmetric"]))]
    per = len(FIELDS) + 5
    while len(row) < 3 * per:
        row += row[:per]
    mom = [x["min_over_max"] for x in v]
    rect = [x["rectangularity"] for x in v]
    sol = [x["solidity"] for x in v]
    row += [min(mom), max(mom), float(np.mean(mom)), max(rect) - min(rect), max(sol) - min(sol),
            max(x["elongation"] for x in v) / max(min(x["elongation"] for x in v), 1e-6),
            max(x["hole_frac"] for x in v), float(len(v))]
    return row


def load_from(path, slot):
    if slot in _CACHE:
        return _CACHE[slot]
    model = None
    if os.path.exists(path):
        try:
            import joblib
            model = joblib.load(path)
        except Exception:
            model = None
    _CACHE[slot] = model
    return model


def load():
    return load_from(MODEL_PATH, "m")


def load_pixels():
    if not USE_PIXELS:
        return None
    if os.environ.get("P2F_DEPTH_HYBRID") == "1":
        return load_from(HYBRID_PATH, "h")
    return load_from(PIXEL_PATH, "p")


def pixel_predict(views, views_events=None):
    model = load_pixels()
    if model is None:
        return None
    from photo2fcstd import embed
    try:
        vector = embed.for_views(views, ALLOW_BACKBONE)
    except Exception:
        return None
    if vector is not None and model.get("kind") == "hybrid":
        try:
            vector = np.hstack([vector, np.array(features(views_events), float)])
        except Exception:
            return None
    if vector is None or len(vector) != model["dims"]:
        return None
    try:
        point = float(model["model"].predict(np.array([vector], float))[0])
    except Exception:
        return None
    off = model["offset"]
    return (float(np.exp(point)), float(np.exp(point - off)), float(np.exp(point + off)),
            1.0 - model.get("alpha", 0.2))


def predict(views):
    """Return (ratio, low, high, coverage) or None. Bounds are conformally calibrated."""
    from photo2fcstd import telemetry
    try:
        events = [telemetry.view_event(v) for v in views]
    except Exception:
        events = None
    from_pixels = pixel_predict(views, events)
    if from_pixels is not None:
        return from_pixels
    model = load()
    if model is None or events is None:
        return None
    try:
        x = np.array([features(events)], float)
    except Exception:
        return None
    try:
        if not isinstance(model, dict):
            return float(np.exp(model.predict(x)[0])), None, None, None
        off = model["offset"]
        return (float(np.exp(model["point"].predict(x)[0])),
                float(np.exp(model["lo"].predict(x)[0] - off)),
                float(np.exp(model["hi"].predict(x)[0] + off)),
                1.0 - model.get("alpha", 0.2))
    except Exception:
        return None


def ratio(views):
    got = predict(views)
    return None if got is None else got[0]
