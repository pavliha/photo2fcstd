import os

import numpy as np

FIELDS = ("elongation", "rectangularity", "solidity", "hole_frac", "min_over_max",
          "ellipse_rms", "stroke_px", "length_px", "stations")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_DEPTH_MODEL", os.path.join(ROOT, "data", "depth_model.joblib"))
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


def load():
    if "m" in _CACHE:
        return _CACHE["m"]
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    _CACHE["m"] = model
    return model


def ratio(views):
    from photo2fcstd import telemetry
    model = load()
    if model is None:
        return None
    try:
        events = [telemetry.view_event(v) for v in views]
        return float(np.exp(model.predict(np.array([features(events)], float))[0]))
    except Exception:
        return None
