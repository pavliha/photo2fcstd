"""Does having an edge-on view in the capture actually make the depth prediction better?

Both heads already read three views - the pixel head's 3,072 dims are three 1,024-vectors - so
"use a second view" is not the open question. The open question is whether an *edge-on* view, where
the depth is directly visible as a width, carries information the model is failing to use. The
hand-written edge-on estimator lost to a constant, but that tested one formula, not whether the
signal is there.

If parts whose capture contains an edge-on view are predicted better, there is headroom. If not,
depth is simply not recoverable from these photographs and the item closes.
"""
import json, os, sys

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    from depth_pixel_train import data, pipeline
    from photo2fcstd import stats
    X, y, g = data(hybrid=False)
    rows = [r for r in json.load(open(os.path.join(ROOT, "data", "depth_rows.json"))) if r.get("ratio", 0) > 0]
    emb = np.load(os.path.join(ROOT, "data", "depth_embeddings.npz"), allow_pickle=True)
    have = set(emb["parts"])
    keep = [r for r in rows if r["part"] in have]
    assert len(keep) == len(y), (len(keep), len(y))

    thin = np.array([min(v.get("min_over_max", 1.0) or 1.0 for v in r["views"]) for r in keep])
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = pipeline().fit(X[tr], y[tr]).predict(X[te])
    err = np.abs(pred - y)
    print("  n=%d parts, held out by group\n" % len(y))
    print("  thinnest view in the capture, against how well depth is predicted:")
    print("    %-26s %6s %12s %12s" % ("thinnest silhouette", "n", "median |err|", "within 2x"))
    edges = [0.0, 0.15, 0.3, 0.5, 0.7, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        k = (thin >= lo) & (thin < hi)
        if k.sum() < 20:
            continue
        print("    %-26s %6d %12.3f %11.0f%%"
              % ("%.2f-%.2f" % (lo, hi), k.sum(), np.median(err[k]), 100 * np.mean(err[k] < np.log(2))))
    edge_on = thin < 0.3
    m, lo_, hi_ = stats.mean_ci(err[~edge_on]) if (~edge_on).sum() else (0, 0, 0)
    m2, lo2, hi2 = stats.mean_ci(err[edge_on]) if edge_on.sum() else (0, 0, 0)
    print("\n  with an edge-on view (thinnest < 0.30, n=%d): mean |err| %.3f [%.3f, %.3f]"
          % (edge_on.sum(), m2, lo2, hi2))
    print("  without one                    (n=%d): mean |err| %.3f [%.3f, %.3f]"
          % ((~edge_on).sum(), m, lo_, hi_))
    d = np.array([1.0 if e else 0.0 for e in edge_on])
    print("  correlation between thinness and error: %.2f" % float(np.corrcoef(thin, err)[0, 1]))
    print("\n  a constant for reference: median |err| %.3f" % float(np.median(np.abs(y - np.median(y)))))


if __name__ == "__main__":
    main()
