"""If the viewpoint tilt could be undone exactly, what is that worth end to end?

The error budget blames 0.212 on capture and says the whole of it is tilt, but that was measured
against the orthographic silhouette rather than against the sketch. This warps each photo's mask
by a grid of rectifying homographies, scores every one, and takes the best - the ceiling for any
tilt estimator, however clever.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402
from photo2fcstd.trace import segment_photo, upright_mask  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
TILTS = (0.0, 8.0, 16.0, 24.0, 32.0)
AZIMUTHS = 6


def unrotate(mask, tilt_deg, az_deg):
    h, w = mask.shape
    if tilt_deg <= 0:
        return mask
    t, a = np.radians(tilt_deg), np.radians(az_deg)
    ax = np.array([np.cos(a), np.sin(a), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
    f = 2.2 * max(h, w)
    C = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    H = C @ R @ np.linalg.inv(C)
    return cv2.warpPerspective(mask.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0


def view_of_mask(mask, name):
    v = analysis.view_from_mask(mask, name)
    return v


def one(part):
    try:
        masks = [upright_mask(segment_photo(p))[0] for p in bench.photos_of(part)[:3]]
        base = max(masks, key=lambda m: int(m.sum()))
        out = {}
        for t in TILTS:
            for k in range(AZIMUTHS if t > 0 else 1):
                az = 180.0 * k / AZIMUTHS
                m = unrotate(base, t, az)
                if m.sum() < 400:
                    continue
                try:
                    doc = spec_mod.assemble([view_of_mask(m, part)], name=part, log=lambda *a: None)
                    out["%.0f_%.0f" % (t, az)] = SS.score_one(doc, IDEAL[part])["region_iou"]
                except Exception:
                    continue
        return part, out
    except Exception:
        return part, None


def main(limit=80):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        res = {k: v for k, v in pool.map(one, parts) if v}
    json.dump(res, open(os.path.join(ROOT, "data", "tilt_ceiling.json"), "w"))
    keys = [k for k, v in res.items() if "0_0" in v]
    base = np.array([res[k]["0_0"] for k in keys])
    best = np.array([max(res[k].values()) for k in keys])
    chosen = [max(res[k], key=res[k].get) for k in keys]
    triv = np.array([SS.trivial_score(IDEAL[k]) for k in keys])
    keen = (1 - triv) >= 0.15
    print("tilt ceiling over %d parts (%d discriminating)\n" % (len(keys), keen.sum()))
    print("  %-32s %8s %8s" % ("", "all", "discriminating"))
    print("  %-32s %8.3f %8.3f" % ("as shot", base.mean(), base[keen].mean()))
    print("  %-32s %8.3f %8.3f" % ("best rectification per part", best.mean(), best[keen].mean()))
    print("  %-32s %8.3f %8.3f" % ("headroom", best.mean() - base.mean(),
                                   best[keen].mean() - base[keen].mean()))
    zero = sum(1 for c in chosen if c == "0_0")
    print("\n  %d of %d parts are already best as shot" % (zero, len(keys)))
    tl = [float(c.split("_")[0]) for c in chosen]
    print("  chosen tilt: median %.0f deg, mean %.1f" % (np.median(tl), np.mean(tl)))


if __name__ == "__main__":
    main()
