"""Train the tolerance selector on the end-to-end score, and report against the ceiling."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from multiprocessing import Pool  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from photo2fcstd import eps_model as EM, sketch_score as SS, stats  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
CEIL = json.load(open(os.path.join(ROOT, "data", "eps_ceiling.json")))


def one(part):
    from photo2fcstd import analysis, bench
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        v = max(views, key=lambda x: x["shape"]["rectangularity"])
        raw = v["shape"]["raw"]
        return part, EM.features_for(v, raw, EM.element_counts(raw, v["length_px"]))
    except Exception:
        return part, None


def main():
    parts = [p for p in CEIL if len(CEIL[p]) == len(EM.CANDIDATES)]
    parts = [p for p in parts if 1 - CEIL[p][str(EM.CANDIDATES[0])]["trivial"] >= 0.15]
    with Pool(6) as pool:
        feats = {k: v for k, v in pool.map(one, parts) if v}
    parts = [p for p in parts if p in feats]
    X = np.array([feats[p] for p in parts], float)
    I = np.array([[CEIL[p][str(e)]["iou"] for e in EM.CANDIDATES] for p in parts], float)
    y = I.argmax(1)
    g = np.arange(len(parts))
    idx = np.arange(len(parts))
    pred = np.zeros(len(parts), int)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4,
                                           min_samples_leaf=12, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    shipped = I[:, EM.CANDIDATES.index(EM.DEFAULT)]
    print("n=%d discriminating parts, held out 5-fold by part\n" % len(parts))
    print("  %-30s %8s" % ("chooser", "IoU"))
    print("  %-30s %8.3f" % ("shipped, one tolerance for all", shipped.mean()))
    print("  %-30s %8.3f" % ("learned per part", I[idx, pred].mean()))
    print("  %-30s %8.3f" % ("best per part (oracle)", I.max(1).mean()))
    m_, lo, hi = stats.mean_ci(I[idx, pred] - shipped)
    print("\n  learned minus shipped: %+.4f [%+.4f, %+.4f]" % (m_, lo, hi))
    print("  picks the best tolerance %.0f%% of the time (chance %.0f%%)"
          % (100 * np.mean(pred == y), 100 / len(EM.CANDIDATES)))
    got = (I[idx, pred].mean() - shipped.mean()) / max(I.max(1).mean() - shipped.mean(), 1e-9)
    print("  captures %.0f%% of the available headroom" % (100 * got))
    if "--save" in sys.argv:
        import joblib
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4,
                                           min_samples_leaf=12, l2_regularization=1.0, random_state=0)
        m.fit(X, y)
        joblib.dump(m, EM.MODEL_PATH)
        print("\nsaved", EM.MODEL_PATH)


if __name__ == "__main__":
    main()
