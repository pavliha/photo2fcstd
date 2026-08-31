"""Fit the depth head on frozen image features, with conformal bounds."""
import json
import sys

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)
ALPHA = 0.2
OUT = "data/depth_pixel_model.joblib"
HYBRID_OUT = "data/depth_hybrid_model.joblib"


def pipeline():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def data(hybrid=False):
    from photo2fcstd import depth_model
    rows = [r for r in json.load(open("data/depth_rows.json")) if r.get("ratio", 0) > 0]
    emb = np.load("data/depth_embeddings.npz", allow_pickle=True)
    vectors = dict(zip(emb["parts"], emb["X"]))
    groups = json.load(open("data/part_groups.json"))
    keep = [r for r in rows if r["part"] in vectors]
    if hybrid:
        keep = [r for r in keep if _shape(depth_model, r) is not None]
        X = np.array([np.hstack([vectors[r["part"]], _shape(depth_model, r)]) for r in keep], float)
    else:
        X = np.array([vectors[r["part"]] for r in keep], float)
    y = np.log([r["ratio"] for r in keep])
    g = np.array([groups.get(r["part"], -1) for r in keep])
    return X, y, g


def conformal_offset(X, y, g):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = pipeline()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    residual = np.abs(pred - y)
    return float(np.quantile(residual, 1 - ALPHA)), pred


def _shape(depth_model, row):
    try:
        return np.array(depth_model.features(row["views"]), float)
    except Exception:
        return None


def main():
    hybrid = "--hybrid" in sys.argv
    X, y, g = data(hybrid)
    offset, pred = conformal_offset(X, y, g)
    covered = np.mean(np.abs(pred - y) <= offset)
    model = pipeline()
    model.fit(X, y)
    out = HYBRID_OUT if hybrid else OUT
    joblib.dump({"kind": "hybrid" if hybrid else "pixels", "model": model, "offset": offset,
                 "alpha": ALPHA, "dims": X.shape[1]}, out)
    print("trained on %d parts x %d dims; conformal offset %.3f covers %.0f%% out of fold; wrote %s"
          % (X.shape[0], X.shape[1], offset, 100 * covered, out))


if __name__ == "__main__":
    main()
