"""Can pixels pick the rectifying warp that silhouette statistics cannot?"""
import json
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from photo2fcstd import bench, embed, stats
from photo2fcstd.trace import segment_photo, upright_mask

ALPHAS = np.logspace(-2, 4, 25)
BASE = "0_0"


def warped_photo(part):
    paths = bench.photos_of(part)[:3]
    if len(paths) < 3:
        return None, None
    areas = [(int(upright_mask(segment_photo(p))[0].sum()), p) for p in paths]
    return max(areas)[1], paths


def data():
    rows = json.load(open("data/tilt_criteria.json"))
    groups = json.load(open("data/part_groups.json"))
    keys = sorted({k for v in rows.values() for k in v})
    parts, X, Y, g = [], [], [], []
    for part, warps in sorted(rows.items()):
        if BASE not in warps:
            continue
        target, paths = warped_photo(part)
        if target is None:
            continue
        try:
            vecs = embed.vectors_for([target] + [p for p in paths if p != target], allow_backbone=False)
        except Exception:
            continue
        if vecs is None:
            continue
        parts.append(part)
        X.append(np.hstack([vecs[0], vecs.mean(axis=0)]))
        Y.append([warps[k]["iou"] if k in warps else np.nan for k in keys])
        g.append(groups.get(part, -1))
    return keys, np.array(X, float), np.array(Y, float), np.array(g), parts


def main():
    keys, X, Y, g, parts = data()
    base_col = keys.index(BASE)
    print("parts %d, warps %d, feature dims %d" % (len(parts), len(keys), X.shape[1]))
    pred = np.full_like(Y, np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, Y[:, base_col], g):
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[tr, j])
            if ok.sum() < 40:
                continue
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr][ok], Y[tr, j][ok])
            pred[te, j] = m.predict(X[te])
    valid = ~np.isnan(Y)
    scores = np.where(np.isnan(pred), -np.inf, np.where(valid, pred, -np.inf))
    picks = scores.argmax(axis=1)
    chosen = np.array([Y[i, picks[i]] for i in range(len(picks))])
    as_shot = Y[:, base_col]
    oracle = np.nanmax(np.where(valid, Y, -np.inf), axis=1)
    keep = ~np.isnan(as_shot) & ~np.isnan(chosen)
    print("as shot        %.3f" % as_shot[keep].mean())
    print("pixels pick    %.3f" % chosen[keep].mean())
    print("oracle         %.3f" % oracle[keep].mean())
    r = stats.paired_delta({i: float(v) for i, v in enumerate(as_shot[keep])},
                           {i: float(v) for i, v in enumerate(chosen[keep])})
    ceiling = oracle[keep].mean() - as_shot[keep].mean()
    print("pixels vs as shot %+.3f [%+.3f, %+.3f]%s   = %.0f%% of the ceiling"
          % (r["delta"], r["lo"], r["hi"], "" if r["significant"] else "  (not resolved)",
             100 * r["delta"] / ceiling if ceiling else 0))
    print("keeps the photo unwarped on %.0f%% of parts" % (100 * np.mean(picks == base_col)))


if __name__ == "__main__":
    main()
