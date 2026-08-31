"""Does view selection do better on pixels than on silhouette statistics?"""
import json
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from photo2fcstd import embed, stats, view_rank
from photo2fcstd.bench import photos_of

ALPHAS = np.logspace(-2, 4, 25)


def rows_with_vectors():
    recs = json.load(open("data/view_features.json"))
    out = []
    for r in recs:
        paths = sorted(photos_of(r["part"]))[:3]
        if len(paths) < 3 or len(r["shots"]) < 3:
            continue
        try:
            vecs = embed.vectors_for(paths, allow_backbone=False)
        except Exception:
            continue
        if vecs is None:
            continue
        out.append({"part": r["part"], "shots": r["shots"], "vecs": vecs})
    return out


def main():
    recs = rows_with_vectors()
    groups = json.load(base := open("data/part_groups.json"))
    base.close()
    g = np.array([groups.get(r["part"], -1) for r in recs])
    n = len(recs)
    print("parts %d with three cached vectors" % n)
    context = np.stack([r["vecs"].mean(axis=0) for r in recs])
    X = np.stack([np.hstack([r["vecs"][i], r["vecs"][i] - context[k]])
                  for k, r in enumerate(recs) for i in range(3)])
    y = np.array([s["iou"] for r in recs for s in r["shots"][:3]])
    gg = np.repeat(g, 3)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, gg):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    scores = pred.reshape(n, 3)
    truth = y.reshape(n, 3)
    picked = truth[np.arange(n), scores.argmax(axis=1)]
    first = truth[:, 0]
    oracle = truth.max(axis=1)
    print("photo 1          %.3f" % first.mean())
    print("pixels ranker    %.3f" % picked.mean())
    print("oracle           %.3f" % oracle.mean())
    r = stats.paired_delta({i: float(v) for i, v in enumerate(first)},
                           {i: float(v) for i, v in enumerate(picked)})
    print("pixels vs photo 1 %+.3f [%+.3f, %+.3f]%s"
          % (r["delta"], r["lo"], r["hi"], "" if r["significant"] else "  (not resolved)"))


if __name__ == "__main__":
    main()
