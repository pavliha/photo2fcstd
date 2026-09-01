"""How much silhouette error can carving take before it stops being worth it?

Every carve number in this repo comes from silhouettes rasterised straight from the truth mesh.
Real masks come from RMBG and have a ragged, spatially correlated boundary. Pose error has already
been swept; this is the other half of the capture term, and it is the half that can be measured
without a single new photograph.

The error model displaces the boundary by a smooth random field: take the signed distance to the
silhouette edge, add correlated noise, re-threshold. That reproduces the way a matting model wanders
along an edge rather than eroding uniformly, and its magnitude is in pixels of boundary displacement.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from axis_data import IDEAL, PERM  # noqa: E402
from photo2fcstd import axis_model, carve as C, carve_check as CC, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
LEVELS = (0.0, 1.0, 2.0, 4.0, 8.0)
BIAS = 0.0


def ragged(mask, px, rng, corr=24.0, bias=BIAS):
    if px <= 0 and bias == 0:
        return mask
    m = mask.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = inside - outside
    n = rng.standard_normal(mask.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), corr)
    n /= max(float(n.std()), 1e-6)
    return (sdf + n * px + bias) > 0


def one(args):
    part, px = args
    try:
        m = trimesh.load(truth_of(part))
        m.apply_translation(-np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        size = float(np.max(m.extents))
        views = CC.poses(16, size * 3.2, (6.0, 18.0, 35.0, 60.0))
        rng = np.random.default_rng(abs(hash(part)) % 2**31)
        masks = [ragged(CC.silhouette(m, v), px, rng) for v in views]
        bounds = [(float(m.bounds[0][i]) - 2, float(m.bounds[1][i]) + 2) for i in range(2)]
        bounds += [(0.0, float(m.bounds[1][2]) + 2)]
        carved = C.carve(views, masks, voxel_mm=max(size * 0.01, 0.15), bounds=bounds)
        if carved is None or len(carved["points_mm"]) < 20:
            return part, px, 0.0, 0
        ax = int(np.argmax(MODEL.predict_proba(axis_model.features(carved))[:, 1]))
        cc = dict(carved)
        cc["points_mm"] = carved["points_mm"][:, PERM[ax]]
        spec = C.spec_from_carve(cc, name=part, axis=2)
        s = SS.score_one(spec, IDEAL[part])
        return part, px, s["region_iou"], int(s["counts_mine"] == s["counts_ideal"])
    except Exception:
        return part, px, 0.0, 0


def main(limit=110):
    strata = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "data", "axis_strata.json")))}
    parts = [p for p, r in strata.items() if not r["circle"] and r["px"] >= 12][:limit]
    with Pool(6) as pool:
        rows = pool.map(one, [(p, px) for px in LEVELS for p in parts])
    by = {}
    for part, px, iou, exact in rows:
        by.setdefault(px, {})[part] = (iou, exact)
    json.dump({str(k): v for k, v in by.items()}, open(os.path.join(ROOT, "data", "matting_sweep.json"), "w"))
    base = np.mean([by[0.0][p][0] for p in parts])
    print("carve under matting error, n=%d real-outline parts, 16 views, exact poses\n" % len(parts))
    print("  %-28s %8s %10s %8s" % ("boundary displacement", "IoU", "vs clean", "exact"))
    for px in LEVELS:
        v = np.mean([by[px][p][0] for p in parts])
        e = 100 * np.mean([by[px][p][1] for p in parts])
        print("  %-28s %8.3f %10s %7.0f%%"
              % ("%.0f px" % px, v, "-" if px == 0 else "%+.3f" % (v - base), e))
    print("\n  the photo path on comparable parts: 0.602")


if __name__ == "__main__":
    main()
