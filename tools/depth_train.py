"""Train the depth regressor and calibrate its interval by conformalised quantile regression.

Point estimate, plus a band with a guaranteed coverage rate. Split three ways: fit the
models on train, calibrate the band on a held-out slice, report on a test slice neither
has seen.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd.depth_model import features  # noqa: E402

ALPHA = 0.2


def fit(X, y, **kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6,
                                         min_samples_leaf=15, random_state=0, **kw).fit(X, y)


def conformal_offset(lo, hi, y, alpha=ALPHA):
    err = np.maximum(lo - y, y - hi)
    n = len(err)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(err, q, method="higher"))


def main(rows_path, out_path):
    rows = json.load(open(rows_path))
    X = np.array([features(r["views"]) for r in rows], float)
    y = np.log([r["ratio"] for r in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    a, b = int(0.6 * len(rows)), int(0.8 * len(rows))
    tr, cal, te = idx[:a], idx[a:b], idx[b:]

    point = fit(X[tr], y[tr], l2_regularization=1.0)
    qlo = fit(X[tr], y[tr], loss="quantile", quantile=ALPHA / 2)
    qhi = fit(X[tr], y[tr], loss="quantile", quantile=1 - ALPHA / 2)
    off = conformal_offset(qlo.predict(X[cal]), qhi.predict(X[cal]), y[cal])

    lo, hi = qlo.predict(X[te]) - off, qhi.predict(X[te]) + off
    cover = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
    width = np.exp(hi - lo)
    e = np.abs(point.predict(X[te]) - y[te])
    print("depth model: %d parts (%d train / %d calibrate / %d test)" % (len(rows), len(tr), len(cal), len(te)))
    print("  point estimate  median |log| %.3f   within 2x %.0f%%" % (np.median(e), 100 * np.mean(e < np.log(2))))
    print("  %d%% interval    coverage %.0f%% (target %d%%)   width median %.1fx" %
          (100 * (1 - ALPHA), 100 * cover, 100 * (1 - ALPHA), np.median(width)))
    print("  usable band (tighter than 2x): %.0f%% of parts" % (100 * np.mean(width < 2.0)))

    import joblib
    full = fit(X, y, l2_regularization=1.0)
    flo = fit(X, y, loss="quantile", quantile=ALPHA / 2)
    fhi = fit(X, y, loss="quantile", quantile=1 - ALPHA / 2)
    joblib.dump({"point": full, "lo": flo, "hi": fhi, "offset": off, "alpha": ALPHA,
                 "n_train": len(rows), "coverage": cover}, out_path)
    print("  wrote %s" % out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "depth_rows.json"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "depth_model.joblib"))
