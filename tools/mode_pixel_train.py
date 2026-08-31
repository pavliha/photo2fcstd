"""Fit one IoU regressor per mode on frozen image features; argmax picks the mode."""
import json

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools_shared import MODES, mode_data

OUT = "data/mode_pixel_model.joblib"
ALPHAS = np.logspace(-2, 4, 25)


def main():
    X, Y, _, rows = mode_data()
    heads = {}
    for j, mode in enumerate(MODES):
        ok = ~np.isnan(Y[:, j])
        if ok.sum() < 50:
            continue
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(X[ok], Y[ok, j])
        heads[mode] = m
    joblib.dump({"kind": "pixels", "heads": heads, "dims": X.shape[1]}, OUT)
    print("trained %d mode heads on %d parts x %d dims; wrote %s"
          % (len(heads), X.shape[0], X.shape[1], OUT))


if __name__ == "__main__":
    main()
