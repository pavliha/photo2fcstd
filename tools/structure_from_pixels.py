"""On the parts where tracing fails, is the sketch structure present in the pixels at all?

Everything learned so far predicts a scalar and lets a hand-written tracer build the sketch.
That tracer can only draw what survives segmentation, which is why most bad sketches have no
good trace from any photograph. Before building a primitive predictor with a rendering loss,
this asks the cheaper question: on exactly those parts, can frozen features predict how many
primitives of each kind the part has, better than the tracer managed to draw?
"""
import json
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from photo2fcstd import bench, embed, stats
from photo2fcstd.bench import photos_of

KINDS = ("line", "arc", "circle")
ALPHAS = np.logspace(-2, 4, 25)


def counts(dct):
    return np.array([float((dct or {}).get(k, 0)) for k in KINDS])


def f1(drawn, wanted):
    matched = float(np.minimum(drawn, wanted).sum())
    d, w = float(drawn.sum()), float(wanted.sum())
    if d <= 0 or w <= 0:
        return 0.0
    p, r = matched / d, matched / w
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/sk10"
    cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    groups = json.load(open("data/part_groups.json"))
    rows = []
    for part, row in bench.sketch_scores(run).items():
        if not row.get("trustworthy") or row.get("region_iou") is None:
            continue
        wanted = counts(row.get("counts_ideal"))
        if wanted.sum() <= 0:
            continue
        paths = sorted(photos_of(part))[:3]
        if len(paths) < 3:
            continue
        try:
            vec = embed.vectors_for(paths, allow_backbone=False)
        except Exception:
            continue
        if vec is None:
            continue
        rows.append({"part": part, "x": vec.reshape(-1), "drawn": counts(row.get("counts_mine")),
                     "wanted": wanted, "region": float(row["region_iou"]),
                     "group": groups.get(part, -1)})
    X = np.array([r["x"] for r in rows], float)
    Y = np.log1p(np.array([r["wanted"] for r in rows], float))
    g = np.array([r["group"] for r in rows])
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y[:, 0], g):
        for j in range(Y.shape[1]):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr], Y[tr, j])
            pred[te, j] = m.predict(X[te])
    guessed = np.maximum(np.expm1(pred), 0.0)
    for label, keep in (("all parts", np.ones(len(rows), bool)),
                        ("where tracing fails", np.array([r["region"] < cutoff for r in rows]))):
        idx = np.flatnonzero(keep)
        if len(idx) < 20:
            continue
        traced = [f1(rows[i]["drawn"], rows[i]["wanted"]) for i in idx]
        pixels = [f1(np.round(guessed[i]), rows[i]["wanted"]) for i in idx]
        print("%-22s n=%4d   tracer %.3f   pixels %.3f" % (label, len(idx), np.mean(traced), np.mean(pixels)))
        r = stats.paired_delta({i: float(v) for i, v in enumerate(traced)},
                               {i: float(v) for i, v in enumerate(pixels)})
        print("   %+0.3f [%+.3f, %+.3f]%s" % (r["delta"], r["lo"], r["hi"],
              "" if r["significant"] else "  - indistinguishable from zero"))


if __name__ == "__main__":
    main()
