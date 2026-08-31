"""Carve with the silhouettes right but the poses wrong, which is what registration error is."""
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
from photo2fcstd import carve as C, carve_check as CC, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
LEVELS = (0.0, 0.5, 1.0, 2.0, 5.0)


def jitter(view, deg, mm, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    dR, _ = cv2.Rodrigues((axis * np.radians(deg * rng.normal())).reshape(3, 1))
    R, _ = cv2.Rodrigues(view["rvec"])
    rvec, _ = cv2.Rodrigues(dR @ R)
    return {**view, "rvec": rvec, "tvec": view["tvec"] + rng.normal(scale=mm, size=(3, 1))}


def one(args):
    part, deg = args
    try:
        m = trimesh.load(truth_of(part))
        m.apply_translation(-np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        size = float(np.max(m.extents))
        views = CC.poses(16, size * 3.2, (6.0, 18.0, 35.0, 60.0))
        masks = [CC.silhouette(m, v) for v in views]
        rng = np.random.default_rng(abs(hash(part)) % 2**31)
        used = views if deg == 0 else [jitter(v, deg, size * 0.005 * deg, rng) for v in views]
        bounds = [(float(m.bounds[0][i]) - 2, float(m.bounds[1][i]) + 2) for i in range(2)]
        bounds += [(0.0, float(m.bounds[1][2]) + 2)]
        carved = C.carve(used, masks, voxel_mm=max(size * 0.01, 0.15), bounds=bounds)
        if carved is None or len(carved["points_mm"]) < 20:
            return part, deg, 0.0
        from photo2fcstd import axis_model
        ax = int(np.argmax(MODEL.predict_proba(axis_model.features(carved))[:, 1]))
        cc = dict(carved)
        cc["points_mm"] = carved["points_mm"][:, PERM[ax]]
        spec = C.spec_from_carve(cc, name=part, axis=2)
        return part, deg, SS.score_one(spec, IDEAL[part])["region_iou"]
    except Exception:
        return part, deg, 0.0


def main():
    strata = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "data", "axis_strata.json")))}
    real = [p for p, r in strata.items() if not r["circle"] and r["px"] >= 12][:150]
    jobs = [(p, d) for d in LEVELS for p in real]
    with Pool(6) as pool:
        out = pool.map(one, jobs)
    by = {}
    for part, deg, v in out:
        by.setdefault(deg, {})[part] = v
    json.dump({str(k): v for k, v in by.items()}, open(os.path.join(ROOT, "data", "pose_noise.json"), "w"))
    print("carve under pose error, %d real-outline parts, 16 views\n" % len(real))
    print("  %-24s %8s %10s" % ("rotation error", "IoU", "vs exact"))
    base = np.mean(list(by[0.0].values()))
    for d in LEVELS:
        v = np.mean([by[d][p] for p in real])
        print("  %-24s %8.3f %10s" % ("%.1f degrees" % d, v, "-" if d == 0 else "%+.3f" % (v - base)))
    print("\n  the photo path on this stratum: 0.590")


if __name__ == "__main__":
    main()
