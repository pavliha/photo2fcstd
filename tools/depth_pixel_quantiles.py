"""Can the pixel depth head say which parts are uncertain, rather than one width for all?

`pixel_predict` returns `exp(point +/- offset)` with a fixed offset, so 46 of 47 measured parts get
exactly a 3.20x band. Marginal coverage holds, but the interval carries no per-part information -
and a constant band printed as a per-part prediction invites a reader to compare parts by it.

This fits proper quantile heads on the same embeddings and conformalises them the CQR way, then
asks two questions: does the width actually vary, and does coverage still hold out of fold. If the
answer to either is no, the honest fix is to stop printing a per-part interval at all.
"""
import json, os, sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
ALPHA = 0.2
COMPONENTS = 64


def quantile_head(q):
    return make_pipeline(StandardScaler(), PCA(n_components=COMPONENTS, random_state=0),
                         GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0,
                                                   n_estimators=200, max_depth=3))


def main():
    from depth_pixel_train import data
    X, y, g = data(hybrid=False)
    print("  %d parts x %d dims, %d groups\n" % (X.shape[0], X.shape[1], len(set(g))))
    lo_p, hi_p, pt_p = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        lo_p[te] = quantile_head(ALPHA / 2).fit(X[tr], y[tr]).predict(X[te])
        hi_p[te] = quantile_head(1 - ALPHA / 2).fit(X[tr], y[tr]).predict(X[te])
        pt_p[te] = quantile_head(0.5).fit(X[tr], y[tr]).predict(X[te])
    score = np.maximum(lo_p - y, y - hi_p)
    off = float(np.quantile(score, 1 - ALPHA))
    lo, hi = lo_p - off, hi_p + off
    width = hi - lo
    covered = float(np.mean((y >= lo) & (y <= hi)))

    from photo2fcstd import depth_model as DM
    shipped = DM.load_pixels()
    s_off = shipped["offset"]
    s_pred = np.zeros_like(y)
    from depth_pixel_train import pipeline
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        s_pred[te] = pipeline().fit(X[tr], y[tr]).predict(X[te])
    s_cov = float(np.mean(np.abs(s_pred - y) <= s_off))

    print("  %-22s %10s %12s %12s %12s" % ("band", "coverage", "median width", "p10 width", "p90 width"))
    print("  %-22s %9.0f%% %11.2fx %11.2fx %11.2fx"
          % ("shipped, constant", 100 * s_cov, np.exp(2 * s_off), np.exp(2 * s_off), np.exp(2 * s_off)))
    print("  %-22s %9.0f%% %11.2fx %11.2fx %11.2fx"
          % ("quantile heads + CQR", 100 * covered, np.exp(np.median(width)),
             np.exp(np.quantile(width, 0.1)), np.exp(np.quantile(width, 0.9))))
    print("\n  claimed coverage %.0f%%" % (100 * (1 - ALPHA)))
    spread = np.exp(np.quantile(width, 0.9)) / max(np.exp(np.quantile(width, 0.1)), 1e-9)
    print("  the new band is %.1fx wider at p90 than p10, so it does%s vary per part"
          % (spread, "" if spread > 1.5 else " NOT"))
    err = np.abs(pt_p - y)
    corr = float(np.corrcoef(width, err)[0, 1])
    print("  correlation between band width and actual error: %.2f%s"
          % (corr, "  <- it knows which parts are hard" if corr > 0.2 else "  <- it does not track difficulty"))
    json.dump({"coverage": covered, "median_width": float(np.exp(np.median(width))),
               "shipped_coverage": s_cov, "shipped_width": float(np.exp(2 * s_off)),
               "corr_width_error": corr}, open(os.path.join(ROOT, "data", "depth_quantiles.json"), "w"))


if __name__ == "__main__":
    main()
