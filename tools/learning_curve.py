"""Would more training data help, or have these heads stopped learning?"""
import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from photo2fcstd import stats

ALPHAS = np.logspace(-2, 4, 25)
FRACTIONS = (0.15, 0.3, 0.5, 0.75, 1.0)


def depth_data():
    rows = [r for r in json.load(open("data/depth_rows.json")) if r.get("ratio", 0) > 0]
    emb = np.load("data/depth_embeddings.npz", allow_pickle=True)
    vec = dict(zip(emb["parts"], emb["X"]))
    groups = json.load(open("data/part_groups.json"))
    keep = [r for r in rows if r["part"] in vec]
    X = np.array([vec[r["part"]] for r in keep], float)
    y = np.log([r["ratio"] for r in keep])
    g = np.array([groups.get(r["part"], -1) for r in keep])
    return X, y, g


def curve(X, y, g, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for frac in FRACTIONS:
        errors = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            take = rng.permutation(tr)[:max(20, int(round(frac * len(tr))))]
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(X[take], y[take])
            errors.extend(np.abs(model.predict(X[te]) - y[te]).tolist())
        out.append((frac, len(take), float(np.mean(errors)),
                    float(1 - np.var(errors) / np.var(y))))
    return out


def main():
    X, y, g = depth_data()
    print("depth head: %d parts, %d features" % X.shape)
    print("  %-8s %-8s %-14s %s" % ("fraction", "n train", "mean |log| err", "R2"))
    rows = curve(X, y, g)
    for frac, n, err, r2 in rows:
        print("  %-8.2f %-8d %-14.4f %.3f" % (frac, n, err, r2))
    gain = rows[-2][2] - rows[-1][2]
    print("\n  last doubling of data bought %.4f of error (%.1f%% of the remaining error)"
          % (gain, 100 * gain / max(rows[-1][2], 1e-9)))


if __name__ == "__main__":
    main()
