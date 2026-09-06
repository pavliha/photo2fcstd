import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
RECS = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
RECS = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))


def one(part):
    from photo2fcstd import bench, multiview, recognise, sketch_score as SS
    photos = bench.photos_of(part)[:3]
    try:
        rec = {**RECS[part], "single_extrusion": True}
        spec = recognise.revolve_spec(photos, rec, name=part) if rec.get("revolve") \
            else recognise.route(photos, name=part, rec=rec)[0]
        sc = SS.score_one(spec, IDEAL[part])
        row = {"part": part, "mode": spec.get("mode"), "region_iou": float(sc["region_iou"]),
               "f1": float((sc.get("primitive_f1") or {}).get("f1", 0.0))}
        spec2, info = multiview.maybe_rectify(photos, spec, name=part)
        cand = spec2 if info.get("applied") else info.get("candidate")
        if cand is not None:
            sc2 = SS.score_one(cand, IDEAL[part])
            row.update({"region_iou_mv": float(sc2["region_iou"]), "f1_mv": float((sc2.get("primitive_f1") or {}).get("f1", 0.0)),
                        "depth_px_mv": cand["outline"]["depth_px"]})
        return {**row, **{k: v for k, v in info.items() if k not in ("mask", "candidate")}}
    except Exception as e:
        return {"part": part, "error": str(e)[:200]}


def main():
    parts = open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split()
    k, n = (int(x) for x in os.environ.get("MV_SHARD", "0/1").split("/"))
    parts = parts[k::n]
    out = os.path.join(ROOT, "runs", os.environ.get("MV_BENCH_OUT", "multiview_bench.json").replace(".json", "_%d.json" % k if n > 1 else ".json"))
    rows = []
    for i, part in enumerate(parts):
        rows.append(one(part))
        r = rows[-1]
        print(i, part, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("region_iou", "region_iou_mv", "applied", "reason", "error")}, flush=True)
        json.dump(rows, open(out, "w"), indent=1)
    ok = [r for r in rows if "region_iou_mv" in r]
    import numpy as np
    print("MV n=%d base %.3f mv %.3f applied %d" % (len(ok), np.mean([r["region_iou"] for r in ok]), np.mean([r["region_iou_mv"] for r in ok]), sum(bool(r.get("applied")) for r in ok)))


if __name__ == "__main__":
    main()
