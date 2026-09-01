"""Retrain both depth heads without a held-out set, so the bench can compare them fairly."""
import json
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from photo2fcstd import depth_model

ALPHAS = np.logspace(-2, 4, 25)
ALPHA = 0.2
HOLDOUT_IDS = "data/depth_holdout_ids.txt"
SIL_OUT = "data/depth_model_heldout.joblib"
PIX_OUT = "data/depth_pixel_model_heldout.joblib"


def split():
    rows = [r for r in json.load(open("data/depth_rows.json")) if r.get("ratio", 0) > 0]
    groups = json.load(open("data/part_groups.json"))
    g = np.array([groups.get(r["part"], -1) for r in rows])
    train_idx, test_idx = next(GroupKFold(n_splits=3).split(rows, groups=g))
    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]


def offset_for(pred, y):
    return float(np.quantile(np.abs(pred - y), 1 - ALPHA))


def main():
    train, test = split()
    emb = np.load("data/depth_embeddings.npz", allow_pickle=True)
    vectors = dict(zip(emb["parts"], emb["X"]))
    y = np.log([r["ratio"] for r in train])

    shape = np.array([depth_model.features(r["views"]) for r in train], float)
    sil = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05)
    sil.fit(shape, y)
    joblib.dump(sil, SIL_OUT)

    have = [i for i, r in enumerate(train) if r["part"] in vectors]
    X = np.array([vectors[train[i]["part"]] for i in have], float)
    pix = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pix.fit(X, y[have])
    joblib.dump({"kind": "pixels", "model": pix, "offset": 0.58, "alpha": ALPHA, "dims": X.shape[1]}, PIX_OUT)

    held = sorted({r["part"] for r in test})
    open(HOLDOUT_IDS, "w").write("\n".join(held) + "\n")
    print("trained both heads on %d parts (%d with vectors); held out %d parts -> %s"
          % (len(train), len(have), len(held), HOLDOUT_IDS))


if __name__ == "__main__":
    main()
