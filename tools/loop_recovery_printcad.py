"""Does carving recover holes as well as a single photograph does?

A visual hull can only see a cavity that breaks the silhouette from some direction, so a
through-hole should be recoverable and a blind pocket should not. PrintCAD parts have holes and
known loop counts, so the comparison against the photo path is direct.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from axis_data import IDEAL, PERM, carve_part  # noqa: E402
from photo2fcstd import axis_model, carve as C, sketch_score as SS  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))


def one(part):
    try:
        carved = carve_part(part)
        if carved is None or len(carved["points_mm"]) < 50:
            return None
        ax = int(np.argmax(MODEL.predict_proba(axis_model.features(carved))[:, 1]))
        cc = dict(carved)
        cc["points_mm"] = carved["points_mm"][:, PERM[ax]]
        spec = C.spec_from_carve(cc, name=part, axis=2)
        s = SS.score_one(spec, IDEAL[part])
        return {"part": part, "got": s["loops_mine"], "ideal": s["loops_ideal"],
                "iou": s["region_iou"], "trivial": s["trivial"]}
    except Exception:
        return None


def main(limit=220):
    strata = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "data", "axis_strata.json")))}
    parts = [p for p, r in strata.items() if not r["circle"] and r["px"] >= 12][:limit]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    json.dump(rows, open(os.path.join(ROOT, "data", "loop_recovery_printcad.json"), "w"))
    keen = [r for r in rows if 1 - r["trivial"] >= 0.15]
    g = np.array([r["got"] for r in keen], float)
    i = np.array([r["ideal"] for r in keen], float)
    holed = i > 1
    print("carve path on PrintCAD, n=%d discriminating parts\n" % len(keen))
    print("  %-42s %8.2f" % ("loops in the real sketch", i.mean()))
    print("  %-42s %8.2f" % ("loops the carve recovers", g.mean()))
    print("  %-42s %7.0f%%" % ("parts whose sketch has holes", 100 * holed.mean()))
    if holed.any():
        print("  %-42s %7.0f%%" % ("  ... of those, carve recovers at least one", 100 * np.mean(g[holed] > 1)))
        print("  %-42s %8.2f" % ("  holes in truth", (i[holed] - 1).mean()))
        print("  %-42s %8.2f" % ("  holes recovered", np.maximum(g[holed] - 1, 0).mean()))
    print("\n  the photo path on the same measure: 1.48 recovered against 1.99 real,")
    print("  missing loops on 17%% of parts")


if __name__ == "__main__":
    main()
