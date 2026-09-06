import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4, min_samples_leaf=15, l2_regularization=1.0, random_state=0).fit(X, y)


def picks(model, rows):
    return np.array([int(np.argmax(model.predict_proba(np.array(r["x"], float))[:, 1])) for r in rows])


def report(name, chosen, rows):
    iou = np.array([r["iou"] for r in rows], float); best = np.array([r["best"] for r in rows])
    print("  %-26s agrees %3.0f%%  region %.3f" % (name, 100 * np.mean(chosen == best), np.mean(iou[np.arange(len(rows)), chosen])))


def main():
    from sklearn.model_selection import GroupKFold
    import joblib
    rows = json.load(open(os.path.join(ROOT, "data", "axis_rows_board.json")))
    bench = set(open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split())
    train = [r for r in rows if r["part"] not in bench]; held = [r for r in rows if r["part"] in bench]
    X = np.array([f for r in train for f in r["x"]], float); y = np.array([int(a == r["best"]) for r in train for a in (0, 1, 2)]); g = np.repeat(np.arange(len(train)), 3)
    cv = np.zeros(len(train), int)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
        for i in np.unique(g[te]):
            cv[i] = int(np.argmax(p[g[te] == i]))
    old = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
    print("train %d parts (5-fold by part):" % len(train))
    report("thinnest extent", np.array([int(np.argmax([f[10] for f in r["x"]])) for r in train]), train)
    report("old model (orbit carves)", picks(old, train), train)
    report("new model (board hulls)", cv, train)
    print("  %-26s agrees 100%%  region %.3f" % ("oracle", np.mean([max(r["iou"]) for r in train])))
    new = fit(X, y)
    print("held-out bench parts %d:" % len(held))
    report("thinnest extent", np.array([int(np.argmax([f[10] for f in r["x"]])) for r in held]), held)
    report("old model (orbit carves)", picks(old, held), held)
    report("new model (board hulls)", picks(new, held), held)
    print("  %-26s agrees 100%%  region %.3f" % ("oracle", np.mean([max(r["iou"]) for r in held])))
    if "--save" in sys.argv:
        joblib.dump(fit(np.array([f for r in rows for f in r["x"]], float), np.array([int(a == r["best"]) for r in rows for a in (0, 1, 2)])), os.path.join(ROOT, "data", "axis_model.joblib"))
        print("saved data/axis_model.joblib trained on all %d parts" % len(rows))


if __name__ == "__main__":
    main()
