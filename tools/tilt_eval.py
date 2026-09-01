"""Does the photograph tell us the tilt, and does it tell us more than the silhouette?

Four arms on identical rows and identical folds, held out by object:

- `constant`   predict the training mean, which is what you get for free
- `shape`      nine hand-written silhouette statistics
- `silhouette` DINOv3 on the binary mask, same crop box as the photo
- `part only`  DINOv3 on the photo with the background zeroed - shading on the part alone
- `photo`      DINOv3 on the whole crop, background included

`part only` is the arm that matters. `photo` beating it means the turntable in the background
carried the tilt, which is true of this dataset and not of a part on a desk.
"""
import os, sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def fold_errors(X, y, groups, folds=5):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((X[tr] - mu) / sd, y[tr])
        out[te] = model.predict((X[te] - mu) / sd)
    return np.abs(out - y)


def angular_errors(X, Y, g, folds=5):
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=folds).split(X, Y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        for j in range(Y.shape[1]):
            pred[te, j] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(
                (X[tr] - mu) / sd, Y[tr, j]).predict((X[te] - mu) / sd)
    p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(p * Y, axis=1)), -1, 1)))


def normal_baseline(Y, g, folds=5):
    out = np.zeros(len(Y))
    for tr, te in GroupKFold(n_splits=folds).split(Y, Y, g):
        mu = Y[tr].mean(0)
        mu = mu / np.linalg.norm(mu)
        out[te] = np.degrees(np.arccos(np.clip(np.abs(Y[te] @ mu), -1, 1)))
    return out


def normals(path="data/tilt_pixels.npz"):
    """The face normal in camera coordinates is what a rectifying homography needs.

    Expressed in the object's own frame instead, the azimuth is arbitrary from one object to the
    next and nothing beats the constant - that is a badly posed target, not a negative result.
    """
    from photo2fcstd import stats
    d = np.load(os.path.join(ROOT, path))
    y, g = d["normal"], d["part"]
    base = normal_baseline(y, g)
    print("  %d views over %d objects, target: face normal in camera coordinates\n" % (len(y), len(set(g))))
    print("  %-12s %10s %10s %12s" % ("arm", "MAE deg", "median", "within 10 deg"))
    print("  %-12s %10.2f %10.2f %11.0f%%" % ("constant", base.mean(), np.median(base), 100 * np.mean(base < 10)))
    errs = {}
    for name, X in (("shape", d["feats"]), ("silhouette", d["sil"]), ("part only", d["obj"])):
        e = angular_errors(X, y, g)
        errs[name] = e
        print("  %-12s %10.2f %10.2f %11.0f%%" % (name, e.mean(), np.median(e), 100 * np.mean(e < 10)))
    print()
    for name, e in errs.items():
        m, lo, hi = stats.mean_ci(base - e)
        print("  %-12s beats the constant by %+.2f deg [%+.2f, %+.2f]" % (name, m, lo, hi))
    m, lo, hi = stats.mean_ci(errs["silhouette"] - errs["part only"])
    print("\n  shading beyond the silhouette: %+.2f deg [%+.2f, %+.2f]" % (m, lo, hi))


def main(path="data/tilt_pixels.npz"):
    d = np.load(os.path.join(ROOT, path))
    y, g = d["tilt"], d["part"]
    print("  %d views over %d objects, tilt %.0f-%.0f deg, sd %.1f deg\n"
          % (len(y), len(set(g)), y.min(), y.max(), y.std()))
    arms = {"constant": None, "shape": d["feats"], "silhouette": d["sil"],
            "part only": d["obj"], "photo": d["rgb"]}
    errs = {}
    print("  %-12s %10s %10s %12s" % ("arm", "MAE deg", "median", "within 10 deg"))
    for name, X in arms.items():
        e = fold_errors(X, y, g) if X is not None else np.abs(
            np.concatenate([np.full(te.shape, y[tr].mean()) - y[te]
                            for tr, te in GroupKFold(n_splits=5).split(y, y, g)]))
        if X is None:
            e = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(y, y, g):
                e[te] = np.abs(y[tr].mean() - y[te])
        errs[name] = e
        print("  %-12s %10.2f %10.2f %11.0f%%" % (name, e.mean(), np.median(e), 100 * np.mean(e < 10)))
    from photo2fcstd import stats
    print()
    for name in ("shape", "silhouette", "part only", "photo"):
        m, lo, hi = stats.mean_ci(errs["constant"] - errs[name])
        print("  %-12s beats the constant by %+.2f deg [%+.2f, %+.2f]" % (name, m, lo, hi))
    m, lo, hi = stats.mean_ci(errs["silhouette"] - errs["part only"])
    print("\n  shading beyond the silhouette:   %+.2f deg [%+.2f, %+.2f]" % (m, lo, hi))
    m, lo, hi = stats.mean_ci(errs["part only"] - errs["photo"])
    print("  background beyond the part:      %+.2f deg [%+.2f, %+.2f]" % (m, lo, hi))


if __name__ == "__main__":
    if "--normal" in sys.argv:
        normals()
    else:
        main(*[a for a in sys.argv[1:] if not a.startswith("--")])
