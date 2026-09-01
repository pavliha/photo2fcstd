"""Can the right simplification tolerance be predicted per part, or only chosen with hindsight?"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from photo2fcstd import embed, stats
from photo2fcstd.bench import photos_of

ALPHAS = np.logspace(-2, 4, 25)


def vectors(parts):
    out = {}
    for part in parts:
        paths = sorted(photos_of(part))[:3]
        if len(paths) < 3:
            continue
        try:
            got = embed.vectors_for(paths, allow_backbone=False)
        except Exception:
            continue
        if got is not None:
            out[part] = got.reshape(-1)
    return out


def main():
    labels = json.load(open("data/eps_labels.json"))
    groups = json.load(open("data/part_groups.json"))
    got = vectors(sorted(labels))
    parts = [p for p in sorted(labels) if p in got]
    grid = sorted({float(k) for r in labels.values() for k in r["scores"]})
    X = np.array([got[p] for p in parts], float)
    Y = np.array([[labels[p]["scores"][str(e)] for e in grid] for p in parts], float)
    g = np.array([groups.get(p, -1) for p in parts])
    print("%d parts with vectors, %d tolerances" % (len(parts), len(grid)))
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y[:, 0], g):
        for j in range(Y.shape[1]):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr], Y[tr, j])
            pred[te, j] = m.predict(X[te])
    chosen = Y[np.arange(len(parts)), pred.argmax(axis=1)]
    fixed = Y.mean(axis=0)
    best_fixed = int(np.argmax(fixed))
    oracle = Y.max(axis=1)
    print("  best fixed  eps %.3f -> %.4f" % (grid[best_fixed], fixed[best_fixed]))
    print("  predicted per part      -> %.4f" % chosen.mean())
    print("  oracle per part         -> %.4f" % oracle.mean())
    r = stats.paired_delta({i: float(v) for i, v in enumerate(Y[:, best_fixed])},
                           {i: float(v) for i, v in enumerate(chosen)})
    print("  predicted vs best fixed: %+.4f [%+.4f, %+.4f]%s"
          % (r["delta"], r["lo"], r["hi"], "" if r["significant"] else "  - indistinguishable from zero"))
    print("  captured %.0f%% of the oracle headroom"
          % (100 * (chosen.mean() - fixed[best_fixed]) / max(oracle.mean() - fixed[best_fixed], 1e-9)))


if __name__ == "__main__":
    main()
