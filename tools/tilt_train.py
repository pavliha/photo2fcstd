"""Fit the tilt head on every T-LESS row and save it with the gate it needs.

The head is a ridge on DINOv3 of the part's pixels, predicting the face normal in camera
coordinates. It also stores the centre of the training embeddings, so `tilt_model` can refuse a
photograph that is nothing like what it was fitted on - the depth head's lesson, which predicted
confidently and wrongly on T-LESS until it was gated.
"""
import os, sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main(data="data/tilt_pixels.npz", out="data/tilt_model.joblib"):
    import joblib
    d = np.load(os.path.join(ROOT, data))
    X, Y, g = d["obj"], d["normal"], d["part"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    heads = [RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((X - mu) / sd, Y[:, j]) for j in range(3)]

    held = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, g):
        m2, s2 = X[tr].mean(0), X[tr].std(0) + 1e-9
        for j in range(3):
            held[te, j] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(
                (X[tr] - m2) / s2, Y[tr, j]).predict((X[te] - m2) / s2)
    p = held / np.maximum(np.linalg.norm(held, axis=1, keepdims=True), 1e-9)
    err = np.degrees(np.arccos(np.clip(np.abs(np.sum(p * Y, axis=1)), -1, 1)))

    centre = X.mean(0)
    centre /= np.linalg.norm(centre)
    dist = 1.0 - X @ centre
    joblib.dump({"mu": mu, "sd": sd, "heads": heads, "centre": centre,
                 "gate": float(np.quantile(dist, 0.95)),
                 "held_out_mae": float(err.mean()), "held_out_p90": float(np.quantile(err, 0.9))},
                os.path.join(ROOT, out))
    print("  fitted on %d views over %d objects" % (len(Y), len(set(g))))
    print("  held out by object: %.2f deg MAE, %.2f deg median, p90 %.2f"
          % (err.mean(), np.median(err), np.quantile(err, 0.9)))
    print("  gate: cosine distance %.3f from the training centre (95th percentile)"
          % np.quantile(dist, 0.95))
    print("  wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
