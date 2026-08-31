"""Label each carved part with the axis whose projection best matches its real sketch."""
import json, os, sys
from multiprocessing import Pool

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import axis_model, carve as C, carve_check as CC, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
PERM = {0: [1, 2, 0], 1: [2, 0, 1], 2: [0, 1, 2]}


def carve_part(part, nviews=16):
    m = trimesh.load(truth_of(part))
    m.apply_translation(-np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
    size = float(np.max(m.extents))
    views = CC.poses(nviews, size * 3.2, (6.0, 18.0, 35.0, 60.0))
    masks = [CC.silhouette(m, v) for v in views]
    bounds = [(float(m.bounds[0][i]) - 2, float(m.bounds[1][i]) + 2) for i in range(2)]
    bounds += [(0.0, float(m.bounds[1][2]) + 2)]
    return C.carve(views, masks, voxel_mm=max(size * 0.01, 0.15), bounds=bounds)


def one(part):
    try:
        carved = carve_part(part)
        if carved is None:
            return None
        scores = []
        for ax in (0, 1, 2):
            cc = dict(carved)
            cc["points_mm"] = carved["points_mm"][:, PERM[ax]]
            try:
                spec = C.spec_from_carve(cc, name=part, axis=2)
                scores.append(SS.score_one(spec, IDEAL[part])["region_iou"] if spec.get("outline") else 0.0)
            except Exception:
                scores.append(0.0)
        if max(scores) <= 0:
            return None
        return {"part": part, "x": axis_model.features(carved).tolist(),
                "iou": scores, "best": int(np.argmax(scores))}
    except Exception:
        return None


def main(limit=None, jobs=6):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = parts[:limit] if limit else parts
    with Pool(jobs) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    out = os.path.join(ROOT, "data", "axis_rows.json")
    json.dump(rows, open(out, "w"))
    best = np.array([r["best"] for r in rows])
    gap = np.mean([max(r["iou"]) - sorted(r["iou"])[-2] for r in rows])
    print("%d parts labelled -> %s" % (len(rows), out))
    print("  best axis is X/Y/Z on %s" % np.bincount(best, minlength=3).tolist())
    print("  mean gap between best and second best axis: %.3f" % gap)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
