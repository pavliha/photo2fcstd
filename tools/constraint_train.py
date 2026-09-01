"""Are the constraint labels learnable at all? Baseline first, as always."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from photo2fcstd import constraints as K  # noqa: E402

rows = json.load(open(os.path.join(ROOT, "data", "constraint_rows.json")))


def edge_feats(els, i, diag):
    e = els[i]
    L = K.edge_length(e) / max(diag, 1e-9)
    a = K.edge_angle(e)
    prev, nxt = els[i - 1], els[(i + 1) % len(els)]
    pa, na = K.edge_angle(prev), K.edge_angle(nxt)
    turn = lambda x, y: abs(((x - y + 90) % 180) - 90) if (x is not None and y is not None) else -1
    return [L, -1 if a is None else min(a, 180 - a), -1 if a is None else abs(a - 90),
            -1 if a is None else abs(a - round(a / 15.0) * 15.0),
            float(e.get("type") == "line"), float(e.get("r", 0.0)) / max(diag, 1e-9),
            turn(a, pa), turn(a, na),
            K.edge_length(prev) / max(diag, 1e-9), K.edge_length(nxt) / max(diag, 1e-9),
            float(len(els)), float(i) / max(len(els), 1)]


def main():
    Xa, ya, ga = [], [], []
    Xp, yp, gp, kindp = [], [], [], []
    for gi, r in enumerate(rows):
        els = r["elements"]
        pts = np.array([e.get("p0", [0, 0]) for e in els] + [e.get("p1", [0, 0]) for e in els], float)
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else 1.0
        f = [edge_feats(els, i, diag) for i in range(len(els))]
        for i, a in enumerate(r["angle"]):
            if a is None or els[i].get("type") != "line":
                continue
            Xa.append(f[i]); ya.append(0 if a == K.HORIZONTAL else 1 if a == K.VERTICAL else 2); ga.append(gi)
        for kind, i, j, v in r["pairs"]:
            if i >= len(f) or j >= len(f):
                continue
            Xp.append(f[i] + f[j] + [abs(f[i][0] - f[j][0]), abs(f[i][1] - f[j][1]), abs(j - i)])
            yp.append(v); gp.append(gi); kindp.append(kind)
    Xa, ya, ga = np.array(Xa, float), np.array(ya), np.array(ga)
    Xp, yp, gp, kindp = np.array(Xp, float), np.array(yp), np.array(gp), np.array(kindp)

    def cv(X, y, g, name):
        if len(np.unique(y)) < 2 or len(X) < 200:
            print("  %-16s too few examples (%d)" % (name, len(X)))
            return
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4,
                                               min_samples_leaf=20, random_state=0)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            pred[te] = p
        base = max(np.mean(y == v) for v in np.unique(y))
        print("  %-16s n=%6d  majority %.3f  model %.3f  %s"
              % (name, len(y), base, np.mean(pred == y),
                 "LEARNS" if np.mean(pred == y) > base + 0.02 else "no better than the baseline"))

    print("held out 5-fold, split by part\n")
    cv(Xa, ya, ga, "angle class")
    for kind in ("equal_length", "parallel", "perpendicular", "equal_radius"):
        s = kindp == kind
        cv(Xp[s], yp[s], gp[s], kind)


if __name__ == "__main__":
    main()
