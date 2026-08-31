"""Per-part silhouette features + the true depth/length ratio from the STEP file."""
import json, os
from multiprocessing import Pool
import numpy as np
from photo2fcstd import analysis, telemetry
from photo2fcstd.bench import photos_of
from photo2fcstd.sketch_score import face_aspect

S = os.path.dirname(os.path.abspath(__file__))
IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))


def face_span(rec):
    p = np.array([q for lp in rec["loops"] for e in lp for q in e.get("xy", [])], float)
    b = p.max(0) - p.min(0)
    return float(max(b))


def one(part):
    try:
        views = [telemetry.view_event(analysis.view(p)) for p in photos_of(part)[:3]]
        rec = IDEAL[part]
        span = face_span(rec)
        if not span or not rec.get("depth"):
            return None
        ratio = rec["depth"] / span
        if not (0.005 < ratio < 3.0):
            return None
        return {"part": part, "views": views, "ratio": ratio}
    except Exception:
        return None


if __name__ == "__main__":
    parts = [p for p in open("data/printcad_all_ids.txt").read().split()
             if p in IDEAL and IDEAL[p].get("depth") and "loops" in IDEAL[p] and face_aspect(IDEAL[p]) >= 0.15]
    with Pool(8) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    json.dump(rows, open(os.path.join(S, "depth_rows_all.json"), "w"))
    r = np.array([x["ratio"] for x in rows])
    print("%d parts   true depth/length: median %.3f  p10 %.3f  p90 %.3f  (log sd %.2f)"
          % (len(rows), np.median(r), *np.percentile(r, [10, 90]), np.std(np.log(r))))
