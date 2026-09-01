"""Retrain the pixel mode selector without a held-out third, so the bench can judge it fairly."""
import json

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools_shared import MODES, mode_data

ALPHAS = np.logspace(-2, 4, 25)
OUT = "data/mode_pixel_model_heldout.joblib"
HOLDOUT_IDS = "data/mode_holdout_ids.txt"


def main():
    X, Y, g, rows = mode_data()
    train_idx, test_idx = next(GroupKFold(n_splits=3).split(X, Y[:, 0], g))
    heads = {}
    for j, mode in enumerate(MODES):
        ok = ~np.isnan(Y[train_idx, j])
        if ok.sum() < 50:
            continue
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(X[train_idx][ok], Y[train_idx, j][ok])
        heads[mode] = m
    joblib.dump({"kind": "pixels", "heads": heads, "dims": X.shape[1]}, OUT)
    held = sorted({rows[i]["part"] for i in test_idx})
    open(HOLDOUT_IDS, "w").write("\n".join(held) + "\n")
    print("trained %d heads on %d parts; held out %d -> %s" % (len(heads), len(train_idx), len(held), HOLDOUT_IDS))


if __name__ == "__main__":
    main()
