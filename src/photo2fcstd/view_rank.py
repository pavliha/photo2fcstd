import json
import os

import numpy as np

from photo2fcstd.settings import PACKAGE_ROOT

KEYS = ("area", "rect", "solidity", "elongation", "symmetry", "hole_frac", "ellipse_rms", "stroke")
MODEL_PATH = os.environ.get("P2F_VIEW_RANKER", os.path.join(PACKAGE_ROOT, "data", "view_ranker.joblib"))
_CACHED = {}


def shot_of(view):
    shape = view["shape"]
    width, height = shape["bbox"]
    return {"area": float(width * height), "rect": float(shape["rectangularity"]),
            "solidity": float(shape["solidity"]), "elongation": float(view["elongation"]),
            "symmetry": float(len(view["symmetric"])), "hole_frac": float(shape["hole_frac"]),
            "ellipse_rms": float(shape["ellipse_rms"]),
            "stroke": float(shape["stroke_px"]) / max(float(view["length_px"]), 1.0)}


def features(shots, index):
    own = [shots[index][k] for k in KEYS]
    relative = []
    for k in KEYS:
        values = [s[k] for s in shots]
        top = max(values) or 1.0
        relative += [shots[index][k] / top if top else 0.0, shots[index][k] - min(values),
                     float(shots[index][k] == max(values))]
    return own + relative


def matrix(shots):
    return np.array([features(shots, i) for i in range(len(shots))], float)


def train(rows, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    X, y = [], []
    for row in rows:
        shots = row["shots"]
        for i, shot in enumerate(shots):
            X.append(features(shots, i))
            y.append(shot["iou"])
    model = GradientBoostingRegressor(random_state=seed, n_estimators=250, max_depth=3, learning_rate=0.05)
    model.fit(np.array(X, float), np.array(y, float))
    return model


def rank(views):
    import joblib
    from photo2fcstd import fallback
    if len(views) < 2:
        return None
    if not os.path.exists(MODEL_PATH):
        fallback.note("view_rank", "no model at %s" % MODEL_PATH)
        return None
    if MODEL_PATH not in _CACHED:
        _CACHED[MODEL_PATH] = joblib.load(MODEL_PATH)
    shots = [shot_of(v) for v in views]
    return list(_CACHED[MODEL_PATH].predict(matrix(shots)))


def best(views):
    scores = rank(views)
    return None if scores is None else views[int(np.argmax(scores))]


def main(argv):
    import joblib
    source = argv[0] if argv else os.path.join(PACKAGE_ROOT, "data", "view_features.json")
    out = argv[1] if len(argv) > 1 else MODEL_PATH
    rows = json.load(open(source))
    joblib.dump(train(rows), out)
    print("trained on %d parts (%d views), wrote %s" % (len(rows), sum(len(r["shots"]) for r in rows), out))


def run():
    import sys
    main(sys.argv[1:])
