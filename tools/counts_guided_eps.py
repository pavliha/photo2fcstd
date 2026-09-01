"""Let pixels say how many primitives the part has, then pick the tolerance that draws that many.

Predicting the best tolerance directly captured 7% of its oracle. Predicting the primitive
counts is a much easier target - frozen features reach 0.79 count-F1 against the tracer's 0.62 -
so this uses the counts as the target and lets the tolerance follow.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from photo2fcstd import embed, stats
from photo2fcstd.bench import photos_of

KINDS = ("line", "arc", "circle")
ALPHAS = np.logspace(-2, 4, 25)


def vec(dct):
    return np.array([float((dct or {}).get(k, 0)) for k in KINDS])


def main():
    labels = json.load(open("data/eps_labels.json"))
    groups = json.load(open("data/part_groups.json"))
    grid = sorted({float(k) for r in labels.values() for k in r["scores"]})
    parts, X = [], []
    for part in sorted(labels):
        paths = sorted(photos_of(part))[:3]
        if len(paths) < 3:
            continue
        try:
            got = embed.vectors_for(paths, allow_backbone=False)
        except Exception:
            continue
        if got is None:
            continue
        parts.append(part)
        X.append(got.reshape(-1))
    X = np.array(X, float)
    Y = np.array([[labels[p]["scores"][str(e)] for e in grid] for p in parts], float)
    ideal = np.log1p(np.array([vec(labels[p]["ideal"]) for p in parts], float))
    drawn = np.array([[vec(labels[p]["counts"][str(e)]) for e in grid] for p in parts], float)
    g = np.array([groups.get(p, -1) for p in parts])
    print("%d parts, %d tolerances" % (len(parts), len(grid)))

    pred = np.zeros_like(ideal)
    for tr, te in GroupKFold(n_splits=5).split(X, ideal[:, 0], g):
        for j in range(ideal.shape[1]):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr], ideal[tr, j])
            pred[te, j] = m.predict(X[te])
    target = np.maximum(np.expm1(pred), 0.0)

    distance = np.linalg.norm(np.log1p(drawn) - np.log1p(target)[:, None, :], axis=2)
    picked = Y[np.arange(len(parts)), distance.argmin(axis=1)]
    fixed = Y.mean(axis=0)
    best = int(np.argmax(fixed))
    oracle = Y.max(axis=1)
    print("  best fixed eps %.3f      %.4f" % (grid[best], fixed[best]))
    print("  counts-guided per part   %.4f" % picked.mean())
    print("  per-part oracle          %.4f" % oracle.mean())
    r = stats.paired_delta({i: float(v) for i, v in enumerate(Y[:, best])},
                           {i: float(v) for i, v in enumerate(picked)})
    print("  guided vs best fixed: %+.4f [%+.4f, %+.4f]%s" % (r["delta"], r["lo"], r["hi"],
          "" if r["significant"] else "  - indistinguishable from zero"))
    print("  captured %.0f%% of the headroom" % (100 * (picked.mean() - fixed[best]) /
                                                 max(oracle.mean() - fixed[best], 1e-9)))


if __name__ == "__main__":
    main()
