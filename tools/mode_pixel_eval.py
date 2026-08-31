"""Does mode selection do better on pixels than on silhouette statistics?"""
import json
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from photo2fcstd import embed, stats
from photo2fcstd.bench import photos_of

MODES = ("stations", "profile", "plan", "revolve")
ALPHAS = np.logspace(-2, 4, 25)


def vector_for(row):
    found = {os.path.basename(p): p for p in photos_of(row["part"])}
    names = [os.path.basename(v["source"]) for v in sorted(row["views"], key=lambda v: -v["elongation"])]
    paths = [found[n] for n in names if n in found][:3]
    if len(paths) < 3:
        return None
    try:
        got = embed.vectors_for(paths, allow_backbone=False)
    except Exception:
        return None
    return None if got is None else got.reshape(-1)


def data():
    rows = json.load(open("data/mode_labels_full.json"))
    groups = json.load(open("data/part_groups.json"))
    kept, X = [], []
    for row in rows:
        if not row.get("iou_per_mode"):
            continue
        v = vector_for(row)
        if v is None:
            continue
        kept.append(row)
        X.append(v)
    Y = np.array([[r["iou_per_mode"].get(m, np.nan) for m in MODES] for r in kept], float)
    g = np.array([groups.get(r["part"], -1) for r in kept])
    return np.array(X, float), Y, g, kept


def choose(scores, Y):
    allowed = ~np.isnan(Y)
    masked = np.where(allowed, scores, -np.inf)
    picks = masked.argmax(axis=1)
    return np.array([Y[i, picks[i]] for i in range(len(picks))]), picks


def main():
    X, Y, g, rows = data()
    print("parts %d with pixel vectors, %d modes" % (len(rows), Y.shape[1]))
    filled = np.nan_to_num(Y, nan=0.0)
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, filled[:, 0], g):
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[tr, j])
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr][ok], Y[tr, j][ok])
            pred[te, j] = m.predict(X[te])
    learned, picks = choose(pred, Y)
    always = {m: np.nanmean(Y[:, i]) for i, m in enumerate(MODES)}
    best_fixed = max(always, key=always.get)
    fixed = Y[:, MODES.index(best_fixed)]
    oracle = np.nanmax(np.where(np.isnan(Y), -np.inf, Y), axis=1)
    print("always %-8s %.3f" % (best_fixed, np.nanmean(fixed)))
    print("pixels argmax    %.3f" % learned.mean())
    print("oracle           %.3f" % oracle.mean())
    ok = ~np.isnan(fixed)
    before = {i: float(v) for i, v in enumerate(fixed[ok])}
    after = {i: float(v) for i, v in enumerate(learned[ok])}
    r = stats.paired_delta(before, after)
    print("pixels vs always %-8s %+.3f [%+.3f, %+.3f]%s"
          % (best_fixed, r["delta"], r["lo"], r["hi"], "" if r["significant"] else "  (not resolved)"))
    share = {m: int((picks == i).sum()) for i, m in enumerate(MODES)}
    print("picks:", share)


if __name__ == "__main__":
    main()
