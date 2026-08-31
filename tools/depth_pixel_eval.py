"""Does the depth head do better on pixels than on silhouette statistics?"""
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from photo2fcstd import depth_model, stats

ALPHAS = np.logspace(-2, 4, 25)


def data():
    rows = [r for r in json.load(open("data/depth_rows.json")) if r.get("ratio", 0) > 0]
    emb = np.load("data/depth_embeddings.npz", allow_pickle=True)
    vectors = {p: v for p, v in zip(emb["parts"], emb["X"])}
    groups_map = json.load(open("data/part_groups.json"))
    keep = [r for r in rows if r["part"] in vectors]
    shape, pixels, y, g = [], [], [], []
    for r in keep:
        try:
            f = depth_model.features(r["views"])
        except Exception:
            continue
        shape.append(f)
        pixels.append(vectors[r["part"]])
        y.append(np.log(r["ratio"]))
        g.append(groups_map.get(r["part"], -1))
    return np.array(shape, float), np.array(pixels, float), np.array(y), np.array(g)


def models():
    return {
        "silhouette + GBM": lambda: GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05),
        "pixels + ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "pixels + PCA + GBM": lambda: make_pipeline(StandardScaler(), PCA(n_components=96, random_state=0),
                                                    GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05)),
    }


def out_of_fold(make, X, y, g):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def line(name, pred, y):
    err = np.abs(pred - y)
    m, lo, hi = stats.mean_ci(err.tolist())[:3]
    return ("%-22s median x%.2f  mean |log| %.3f [%.3f, %.3f]  within 20%%: %2.0f%%  R2 %.3f"
            % (name, np.exp(np.median(err)), m, lo, hi, 100 * np.mean(err < 0.182), 1 - np.var(pred - y) / np.var(y)))


def main():
    shape, pixels, y, g = data()
    print("parts %d  silhouette dims %d  pixel dims %d" % (len(y), shape.shape[1], pixels.shape[1]))
    preds = {}
    for name, make in models().items():
        X = shape if name.startswith("silhouette") else pixels
        preds[name] = out_of_fold(make, X, y, g)
        print(line(name, preds[name], y))
    both = np.hstack([shape, pixels])
    preds["both + ridge"] = out_of_fold(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)), both, y, g)
    print(line("both + ridge", preds["both + ridge"], y))
    base = np.abs(preds["silhouette + GBM"] - y)
    for name, pred in preds.items():
        if name == "silhouette + GBM":
            continue
        after = {i: float(v) for i, v in enumerate(np.abs(pred - y))}
        before = {i: float(v) for i, v in enumerate(base)}
        r = stats.paired_delta(before, after)
        print("  %-22s changes mean |log| error by %+.3f [%+.3f, %+.3f]%s"
              % (name, r["delta"], r["lo"], r["hi"], "" if r["significant"] else "  (not resolved)"))


if __name__ == "__main__":
    main()
