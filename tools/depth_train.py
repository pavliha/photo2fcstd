"""Predict log(depth/length) from silhouette features. Compare to the shipped estimator."""
import json, os
import numpy as np
from photo2fcstd import thresholds as th

S = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("elongation", "rectangularity", "solidity", "hole_frac", "min_over_max",
          "ellipse_rms", "stroke_px", "length_px", "stations")


def features(views):
    v = sorted(views, key=lambda x: -x["elongation"])[:3]
    row = []
    for x in v:
        row += [float(x[f]) for f in FIELDS]
        row += [float(x["ellipse_aspect"] or 0.0), float(x["round"]), float(x["roundish"]),
                float(x["stroke_px"]) / max(float(x["length_px"]), 1.0), float(len(x["symmetric"]))]
    per = len(FIELDS) + 5
    while len(row) < 3 * per:
        row += row[:per]
    mom = [x["min_over_max"] for x in v]
    rect = [x["rectangularity"] for x in v]
    sol = [x["solidity"] for x in v]
    row += [min(mom), max(mom), float(np.mean(mom)), max(rect) - min(rect), max(sol) - min(sol),
            max(x["elongation"] for x in v) / max(min(x["elongation"] for x in v), 1e-6),
            max(x["hole_frac"] for x in v), float(len(v))]
    return row


def shipped_estimate(views):
    v = sorted(views, key=lambda x: -x["elongation"])[:3]
    src = v[0]
    aspect = min(x["min_over_max"] for x in v)
    return th.DEPTH_FROM_ASPECT * aspect


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    rows = json.load(open(os.path.join(S, "depth_rows.json")))
    X = np.array([features(r["views"]) for r in rows], float)
    y = np.log(np.array([r["ratio"] for r in rows], float))
    base = np.log(np.array([max(shipped_estimate(r["views"]), 1e-6) for r in rows], float))
    const = np.full_like(y, np.median(y))

    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6,
                                          min_samples_leaf=15, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])

    def report(name, p):
        e = np.abs(p - y)
        print("  %-34s median |log| %.3f   within 1.5x %2.0f%%   within 2x %2.0f%%"
              % (name, np.median(e), 100 * np.mean(e < np.log(1.5)), 100 * np.mean(e < np.log(2))))

    print("depth prediction on %d trusted parts (5-fold CV)" % len(rows))
    report("best constant (median ratio)", const)
    report("shipped: 0.46 x edge-on aspect", base)
    report("learned: gradient boosting", pred)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=6,
                                      min_samples_leaf=15, l2_regularization=1.0, random_state=0).fit(X, y)
    import joblib
    joblib.dump(m, os.path.join(S, "depth_model.joblib"))
    print("  saved depth_model.joblib")


if __name__ == "__main__":
    main()
