"""Split the carve result by face resolution and by shape, to see where the score comes from."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from axis_data import IDEAL, PERM, carve_part  # noqa: E402
from photo2fcstd import carve as C, sketch_score as SS  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))


def one(r):
    try:
        ax = int(np.argmax(MODEL.predict_proba(np.array(r["x"], float))[:, 1]))
        carved = carve_part(r["part"])
        cc = dict(carved)
        cc["points_mm"] = carved["points_mm"][:, PERM[ax]]
        spec = C.spec_from_carve(cc, name=r["part"], axis=2)
        grid = C.occupancy(cc, axis=2)
        loops = spec["outline"]["loops"]
        return {"part": r["part"], "px": int(min(grid.shape)),
                "circle": bool(len(loops) == 1 and loops[0]["type"] == "circle"),
                "loops": len(loops), "ideal_loops": len(SS.ideal_rings(IDEAL[r["part"]])),
                "iou": r["iou"][ax], "thin": r["iou"][int(np.argmax([f[10] for f in r["x"]]))]}
    except Exception:
        return None


def main():
    rows = json.load(open(os.path.join(ROOT, "data", "axis_holdout.json")))
    with Pool(6) as pool:
        out = [x for x in pool.map(one, rows) if x]
    json.dump(out, open(os.path.join(ROOT, "data", "axis_strata.json"), "w"))
    px = np.array([r["px"] for r in out])
    circ = np.array([r["circle"] for r in out])
    iou = np.array([r["iou"] for r in out])
    thin = np.array([r["thin"] for r in out])
    print("%d parts\n" % len(out))
    print("  %-34s %5s %8s %8s" % ("stratum", "n", "learned", "thinnest"))
    for name, sel in (("everything", np.ones(len(out), bool)),
                      ("traced as a single circle", circ),
                      ("a real outline (not one circle)", ~circ),
                      ("face under 12 px across", px < 12),
                      ("face 12 px or wider", px >= 12),
                      ("wider than 12 px and not a circle", (px >= 12) & ~circ)):
        if sel.sum():
            print("  %-34s %5d %8.3f %8.3f" % (name, sel.sum(), iou[sel].mean(), thin[sel].mean()))
    print("\n  single circles are %.0f%% of parts and carry %.0f%% of the total score" %
          (100 * circ.mean(), 100 * iou[circ].sum() / iou.sum()))
    print("  median face resolution: %d px" % np.median(px))


if __name__ == "__main__":
    main()
