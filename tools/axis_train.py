import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

rows = json.load(open(os.path.join(ROOT, "data", "axis_rows.json")))
X = np.array([f for r in rows for f in r["x"]], float)
y = np.array([int(a == r["best"]) for r in rows for a in (0, 1, 2)])
g = np.repeat(np.arange(len(rows)), 3)
iou = np.array([r["iou"] for r in rows], float)
thinnest = np.array([int(np.argmax([f[10] for f in r["x"]])) for r in rows])

folds = list(GroupKFold(n_splits=5).split(X, y, g))
pred = np.zeros(len(rows), int)
for tr, te in folds:
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4,
                                       min_samples_leaf=15, l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    parts = np.unique(g[te])
    for i in parts:
        pred[i] = int(np.argmax(p[g[te] == i]))

got = lambda c: (float(np.mean(c == np.array([r["best"] for r in rows]))),
                 float(np.mean(iou[np.arange(len(rows)), c])))
print("held-out over %d parts (5-fold, split by part)\n" % len(rows))
print("  %-22s %8s %8s" % ("rule", "agrees", "IoU"))
for name, c in (("thinnest extent", thinnest), ("learned", pred)):
    a, v = got(c)
    print("  %-22s %7.0f%% %8.3f" % (name, 100 * a, v))
print("  %-22s %7.0f%% %8.3f" % ("oracle", 100, iou.max(axis=1).mean()))
print("  %-22s %7s %8.3f" % ("worst case", "", iou.min(axis=1).mean()))

fixed = int(np.sum((pred == np.array([r["best"] for r in rows])) & (thinnest != np.array([r["best"] for r in rows]))))
broke = int(np.sum((pred != np.array([r["best"] for r in rows])) & (thinnest == np.array([r["best"] for r in rows]))))
print("\n  vs thinnest extent: fixed %d parts, broke %d" % (fixed, broke))

if "--save" in sys.argv:
    import joblib
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4,
                                       min_samples_leaf=15, l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    out = os.path.join(ROOT, "data", "axis_model.joblib")
    joblib.dump(m, out)
    print("\nsaved", out)
