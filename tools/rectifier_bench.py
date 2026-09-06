import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def one(part):
    from photo2fcstd import bench, design, recognise, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    photos = bench.photos_of(part)[:3]
    try:
        rec = {**recs[part], "single_extrusion": True}
        if rec.get("revolve"):
            return {"part": part, "skipped": "revolve"}
        spec = recognise.route(photos, name=part, rec=rec)[0]
        sc = SS.score_one(spec, ideal[part])
        row = {"part": part, "mode": spec.get("mode"), "region_iou": float(sc["region_iou"]), "f1": float((sc.get("primitive_f1") or {}).get("f1", 0.0))}
        spec2, info = design._rectifier().maybe_rectify(photos, spec, name=part)
        if info.get("applied"):
            sc2 = SS.score_one(spec2, ideal[part])
            row.update({"region_iou_rect": float(sc2["region_iou"]), "f1_rect": float((sc2.get("primitive_f1") or {}).get("f1", 0.0))})
        return {**row, **{k: v for k, v in info.items() if k != "mask"}}
    except Exception as e:
        return {"part": part, "error": "%s: %s" % (type(e).__name__, str(e)[:120])}


def main():
    parts = open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split()
    with Pool(int(os.environ.get("RECT_WORKERS", "4"))) as pool:
        rows = pool.map(one, parts)
    out = os.path.join(ROOT, "runs", os.environ.get("RECT_BENCH_OUT", "rectifier_bench.json"))
    json.dump(rows, open(out, "w"), indent=1)
    ok = [r for r in rows if "region_iou" in r]; ap = [r for r in ok if r.get("applied")]
    print("RECTIFIER %s: n=%d applied=%d  base %.3f -> rectified %.3f on applied (wins %d losses %d)  | all parts picked %.3f vs base %.3f" % (
        os.environ.get("P2F_RECTIFIER", "vanish"), len(ok), len(ap), np.mean([r["region_iou"] for r in ap]) if ap else 0, np.mean([r["region_iou_rect"] for r in ap]) if ap else 0,
        sum(r["region_iou_rect"] > r["region_iou"] + 0.02 for r in ap), sum(r["region_iou_rect"] < r["region_iou"] - 0.02 for r in ap),
        np.mean([r.get("region_iou_rect", r["region_iou"]) for r in ok]), np.mean([r["region_iou"] for r in ok])))
    reasons = {}
    for r in ok:
        if not r.get("applied"): reasons[r.get("reason", "?")[:40]] = reasons.get(r.get("reason", "?")[:40], 0) + 1
    print("  not applied:", sorted(reasons.items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    main()
