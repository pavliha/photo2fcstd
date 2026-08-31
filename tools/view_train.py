import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from photo2fcstd import view_model as VM  # noqa: E402

res = json.load(open(os.path.join(ROOT, "data", "view_ceiling.json")))
import photo2fcstd.sketch_score as SS  # noqa: E402
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))

rows = []
for part, r in res.items():
    pv = [p for p in r["per_view"] if p]
    if len(pv) < 2:
        continue
    rows.append({"part": part, "x": VM.features(pv).tolist(),
                 "iou": [p["iou"] for p in pv], "shipped": r["shipped"],
                 "keen": 1 - SS.trivial_score(IDEAL[part]) >= 0.15})

X = np.array([f for r in rows for f in r["x"]], float)
y = np.array([int(i == int(np.argmax(r["iou"]))) for r in rows for i in range(len(r["iou"]))])
g = np.array([i for i, r in enumerate(rows) for _ in r["iou"]])

pred = np.zeros(len(rows), int)
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                       min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    for i in np.unique(g[te]):
        pred[i] = int(np.argmax(p[g[te] == i]))

keen = np.array([r["keen"] for r in rows])
get = lambda f: float(np.mean([f(r, i) for r, i in zip(np.array(rows)[keen], pred[keen])]))
print("%d parts, %d discriminating, held out 5-fold by part\n" % (len(rows), keen.sum()))
print("  %-28s %8s" % ("chooser", "IoU"))
print("  %-28s %8.3f" % ("what ships today", get(lambda r, i: r["shipped"])))
print("  %-28s %8.3f" % ("first photo", get(lambda r, i: r["iou"][0])))
print("  %-28s %8.3f" % ("learned", get(lambda r, i: r["iou"][i])))
print("  %-28s %8.3f" % ("best of three (oracle)", get(lambda r, i: max(r["iou"]))))
agree = np.mean([pred[j] == int(np.argmax(rows[j]["iou"])) for j in np.flatnonzero(keen)])
print("\n  agrees with the best view %.0f%% of the time (chance is %.0f%%)"
      % (100 * agree, 100 * np.mean([1.0 / len(r["iou"]) for r in rows])))

if "--save" in sys.argv:
    import joblib
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                       min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    joblib.dump(m, VM.MODEL_PATH)
    print("\nsaved", VM.MODEL_PATH)
