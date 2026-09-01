import os

import numpy as np

from photo2fcstd.settings import PACKAGE_ROOT

MODEL_PATH = os.environ.get("P2F_MODE_PIXEL_MODEL", os.path.join(PACKAGE_ROOT, "data", "mode_pixel_model.joblib"))
ENABLED = os.environ.get("P2F_MODE_PIXELS", "1") == "1"
ALLOW_BACKBONE = os.environ.get("P2F_EMBED_BACKBONE", "1") == "1"
_CACHE = {}


def load():
    if "m" not in _CACHE:
        from photo2fcstd import fallback
        model = None
        if not os.path.exists(MODEL_PATH):
            fallback.note("mode_pixels", "no model at %s" % MODEL_PATH)
        else:
            try:
                import joblib
                model = joblib.load(MODEL_PATH)
            except Exception as exc:
                fallback.note("mode_pixels", "could not load %s: %s" % (MODEL_PATH, exc))
        _CACHE["m"] = model
    return _CACHE["m"]


def scores(views):
    model = load()
    if model is None:
        return None
    from photo2fcstd import embed
    try:
        vector = embed.for_views(views, ALLOW_BACKBONE)
    except Exception as exc:
        from photo2fcstd import fallback
        fallback.note("mode_pixels", str(exc))
        return None
    if vector is None or len(vector) != model["dims"]:
        return None
    x = np.array([vector], float)
    try:
        return {mode: float(head.predict(x)[0]) for mode, head in model["heads"].items()}
    except Exception as exc:
        from photo2fcstd import fallback
        fallback.note("mode_pixels", str(exc))
        return None


def predict(views, allowed):
    if not ENABLED:
        return None
    got = scores(views)
    if not got:
        return None
    usable = {m: s for m, s in got.items() if m in allowed}
    return max(usable, key=usable.get) if usable else None
