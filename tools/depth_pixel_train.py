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


def pipeline():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def data():
    rows = [r for r in json.load(open("data/depth_rows.json")) if r.get("ratio", 0) > 0]
    emb = np.load("data/depth_embeddings.npz", allow_pickle=True)
    vectors = dict(zip(emb["parts"], emb["X"]))
    groups = json.load(open("data/part_groups.json"))
    keep = [r for r in rows if r["part"] in vectors]
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


def main():
    X, y, g = data()
    offset, pred = conformal_offset(X, y, g)
    covered = np.mean(np.abs(pred - y) <= offset)
    model = pipeline()
    model.fit(X, y)
    joblib.dump({"kind": "pixels", "model": model, "offset": offset, "alpha": ALPHA, "dims": X.shape[1]}, OUT)
    print("trained on %d parts x %d dims; conformal offset %.3f covers %.0f%% out of fold; wrote %s"
          % (X.shape[0], X.shape[1], offset, 100 * covered, OUT))


if __name__ == "__main__":
    main()
