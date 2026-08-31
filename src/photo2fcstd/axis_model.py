"""Pick which way to look at a carved volume to see the part's face.

Four hand-written rules plateau at 71% agreement with the best choice, because a plate
is extruded along its short axis and a rod along its long one. This scores each of the
three axes independently and takes the best, which gives three training rows per part
and does not care about axis order.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_AXIS_MODEL", os.path.join(ROOT, "data", "axis_model.joblib"))
_CACHE = {}


def slice_counts(pts, voxel, axis):
    k = np.round(pts[:, axis] / max(voxel, 1e-9)).astype(int)
    _, counts = np.unique(k, return_counts=True)
    return counts


def features_for_axis(carved, axis):
    from photo2fcstd import carve as C
    pts = np.asarray(carved["points_mm"], float)
    voxel = float(carved["voxel_mm"])
    extents = np.array([np.ptp(pts[:, a]) for a in (0, 1, 2)])
    big = max(extents.max(), 1e-9)
    grid = C.occupancy(carved, axis=axis)
    area = float(grid.sum())
    h, w = grid.shape
    counts = slice_counts(pts, voxel, axis)
    trimmed = counts[1:-1] if len(counts) > 4 else counts
    others = [extents[a] for a in (0, 1, 2) if a != axis]
    return [
        extents[axis] / big,                                   # how long this way
        min(others) / big, max(others) / big,                  # the face's own proportions
        extents[axis] / max(min(others), 1e-9),                # thin or long relative to the face
        area / max(len(pts), 1),                               # projected area vs volume
        area / max(h * w, 1),                                  # how much of its box it fills
        min(h, w) / max(max(h, w), 1),                         # aspect of the projection
        float(np.median(trimmed) / max(trimmed.max(), 1)),     # is the section constant
        float(trimmed.std() / max(trimmed.mean(), 1e-9)),      # ... and how much it varies
        len(counts) * voxel / big,                             # slices along this axis
        float(np.argmin(extents) == axis),                     # the old thinnest-extent rule
        float(np.argmax(extents) == axis),
    ]


def features(carved):
    return np.array([features_for_axis(carved, a) for a in (0, 1, 2)], float)


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


def predict_axis(carved):
    """Best axis to project along, or None when no model is available."""
    model = load()
    if model is None:
        return None
    try:
        scores = model.predict_proba(features(carved))[:, 1]
        return int(np.argmax(scores))
    except Exception:
        return None
