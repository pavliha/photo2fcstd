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


CENTRE_PATH = os.path.join(ROOT, "data", "embed_centre.npz")
GATE = os.environ.get("P2F_EMBED_GATE", "1") == "1"


def embedding_is_familiar(vector):
    """Is this embedding anywhere near the images the head was fitted on?

    DINOv3 vectors are L2 normalised, so norm says nothing. Cosine distance to the centre of the
    training photographs does: PrintCAD sits at a median 0.371 from its own centre and T-LESS at
    0.839, with **no** T-LESS image inside PrintCAD's 95th percentile. The pixel head is not wrong
    about those parts so much as never asked - it extrapolates into a region it never saw, which is
    why it predicts 0.108 to 0.480 where the truth runs 0.377 to 1.561. Out there the tabular model
    is the better answer, 0.236 against 1.117.
    """
    if not GATE or not os.path.exists(CENTRE_PATH):
        return True
    try:
        ref = _CACHE.get("centre")
        if ref is None:
            ref = dict(np.load(CENTRE_PATH))
            _CACHE["centre"] = ref
        v = np.asarray(vector, float)
        per = v.reshape(-1, len(ref["centre"])) if v.size % len(ref["centre"]) == 0 else v.reshape(1, -1)
        per = per / np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-9)
        d = float(np.median(1.0 - per @ np.asarray(ref["centre"], float)))
        return d <= float(ref["p95"]) * 1.1
    except Exception:
        return True


def _give_up(reason):
    from photo2fcstd import fallback
    fallback.note("depth_model", reason)
    return None


def load_from(path, slot):
    if slot in _CACHE:
        return _CACHE[slot]
    model = None
    if os.path.exists(path):
        try:
            import joblib
            model = joblib.load(path)
        except Exception as exc:
            _give_up("could not load %s: %s" % (path, exc))
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
    except Exception as exc:
        return _give_up("embedding failed: %s" % exc)
    if vector is not None and model.get("kind") == "hybrid":
        try:
            vector = np.hstack([vector, np.array(features(views_events), float)])
        except Exception as exc:
            return _give_up("hybrid feature join failed: %s" % exc)
    if vector is None:
        return _give_up("no embedding for these views, falling through to the tabular model")
    if len(vector) != model["dims"]:
        return _give_up("embedding is %d dims, model wants %d" % (len(vector), model["dims"]))
    if not embedding_is_familiar(vector):
        return _give_up("these photographs sit outside the range the pixel head was fitted on")
    try:
        point = float(model["model"].predict(np.array([vector], float))[0])
    except Exception as exc:
        return _give_up("pixel model failed: %s" % exc)
    off = model["offset"]
    return (float(np.exp(point)), float(np.exp(point - off)), float(np.exp(point + off)),
            1.0 - model.get("alpha", 0.2), False)


def predict(views):
    """Return (ratio, low, high, coverage, per_part) or None.

    `per_part` says whether the band was predicted for this part or is a fixed calibration applied
    to every part. The pixel head has no quantile heads - it returns `exp(point +/- offset)` - so its
    band is the same 3.20x width for everything, and 46 of 47 measured parts reported exactly that.
    Marginal coverage still holds, but a constant width printed as a per-part interval invites the
    reader to compare parts by it, so the caller must say which it has.

    Fitting real quantile heads on the same embeddings was measured and lost: same 80% coverage,
    median width 6.05x against 3.20x, and width correlating only 0.22 with actual error.
    """
    from photo2fcstd import telemetry
    try:
        events = [telemetry.view_event(v) for v in views]
    except Exception as exc:
        _give_up("view telemetry failed: %s" % exc)
        events = None
    from_pixels = pixel_predict(views, events)
    if from_pixels is not None:
        return from_pixels
    model = load()
    if model is None:
        return _give_up("no tabular model")
    if events is None:
        return _give_up("view telemetry unavailable")
    try:
        x = np.array([features(events)], float)
    except Exception as exc:
        return _give_up("feature build failed: %s" % exc)
    try:
        if not isinstance(model, dict):
            return float(np.exp(model.predict(x)[0])), None, None, None, False
        off = model["offset"]
        return (float(np.exp(model["point"].predict(x)[0])),
                float(np.exp(model["lo"].predict(x)[0] - off)),
                float(np.exp(model["hi"].predict(x)[0] + off)),
                1.0 - model.get("alpha", 0.2), True)
    except Exception as exc:
        return _give_up("tabular model failed: %s" % exc)


def ratio(views):
    got = predict(views)
    return None if got is None else got[0]
