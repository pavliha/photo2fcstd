"""Retrain the view selector on labels the mode router no longer blanks, and compare honestly."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from photo2fcstd import sketch_score as SS, stats, view_model as VM  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
lab = json.load(open(os.path.join(ROOT, "data", "view_ceiling_clean.json")))
rows = [{"part": k, "pv": [p for p in v["per_view"] if p], "shipped": v["shipped"]}
        for k, v in lab.items() if len(v["per_view"]) == 3 and all(v["per_view"])]
rows = [r for r in rows if 1 - SS.trivial_score(IDEAL[r["part"]]) >= 0.15]
SEEN = set(json.load(open(os.path.join(ROOT, "data", "view_ceiling.json"))))
overlap = sum(1 for r in rows if r["part"] in SEEN)
rows = [r for r in rows if r["part"] not in SEEN]
print("dropped %d parts the shipped model trained on, leaving %d\n" % (overlap, len(rows)))

X = np.array([f for r in rows for f in VM.features(r["pv"])], float)
I = np.array([[p["iou"] for p in r["pv"]] for r in rows], float)
y = np.array([int(i == int(np.argmax(r["iou"] if False else [p["iou"] for p in r["pv"]])))
              for r in rows for i in range(3)])
g = np.repeat(np.arange(len(rows)), 3)
idx = np.arange(len(rows))
truth = I.argmax(1)

fresh = np.zeros(len(rows), int)
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                       min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    for i in np.unique(g[te]):
        fresh[i] = int(np.argmax(p[g[te] == i]))

old = np.array([int(np.argmax(VM.load().predict_proba(VM.features(r["pv"]))[:, 1])) for r in rows])

print("clean labels, n=%d discriminating parts, held out 5-fold by part\n" % len(rows))
print("  %-34s %8s %8s" % ("selector", "agree", "IoU"))
print("  %-34s %8.2f %8.3f" % ("chance", 1 / 3, I.mean()))
print("  %-34s %8s %8.3f" % ("first photo", "-", I[:, 0].mean()))
print("  %-34s %8s %8.3f" % ("what ships (whole pipeline)", "-",
                             np.mean([r["shipped"] for r in rows])))
print("  %-34s %8.2f %8.3f" % ("shipped model, trained dirty", np.mean(old == truth), I[idx, old].mean()))
print("  %-34s %8.2f %8.3f" % ("retrained on clean labels", np.mean(fresh == truth), I[idx, fresh].mean()))
print("  %-34s %8.2f %8.3f" % ("oracle", 1.0, I.max(1).mean()))
d = I[idx, fresh] - I[idx, old]
m_, lo, hi = stats.mean_ci(d)
print("\n  retrained minus shipped: %+.4f [%+.4f, %+.4f]" % (m_, lo, hi))

if "--save" in sys.argv:
    import joblib
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                       min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    m.fit(X, y)
    joblib.dump(m, VM.MODEL_PATH)
    print("\nsaved", VM.MODEL_PATH)
