"""Score the photo path and the carve path on the same parts, so the comparison is fair."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_data import IDEAL  # noqa: E402
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402

OUT = os.path.join(ROOT, "runs", "photo_vs_carve")


def photo_one(part):
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        return part, SS.score_one(doc, IDEAL[part])["region_iou"]
    except Exception:
        return part, None


def main():
    rows = json.load(open(os.path.join(ROOT, "data", "axis_holdout.json")))
    have = set(bench.with_photos([r["part"] for r in rows])[0])
    rows = [r for r in rows if r["part"] in have]
    os.makedirs(OUT, exist_ok=True)
    with Pool(6) as pool:
        photo = dict(pool.map(photo_one, [r["part"] for r in rows]))
    import joblib
    m = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
    pairs = [(photo[r["part"]],
              r["iou"][int(np.argmax(m.predict_proba(np.array(r["x"], float))[:, 1]))],
              r["iou"][int(np.argmax([f[10] for f in r["x"]]))],
              max(r["iou"]))
             for r in rows if photo.get(r["part"]) is not None]
    a = np.array(pairs, float)
    json.dump({"parts": [r["part"] for r in rows], "table": a.tolist()},
              open(os.path.join(OUT, "scores.json"), "w"))
    print("%d parts with both photos and a carve\n" % len(a))
    for i, n in enumerate(("photo path", "carve + thinnest extent", "carve + learned axis", "carve oracle")):
        col = a[:, (0, 2, 1, 3)[i]]
        print("  %-26s %.3f" % (n, col.mean()))
    print("\n  carve beats photo on %.0f%% of parts" % (100 * np.mean(a[:, 1] > a[:, 0])))


if __name__ == "__main__":
    main()
