import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
GOLDEN = os.path.join(ROOT, "data", "regression", "round_parts.json")


def round_parts():
    fam = json.load(open(os.path.join(ROOT, "runs", "template_families.json")))
    bench_parts = set(open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split())
    return sorted(p for k in ("circle", "circle + 1xcircle") for p in fam[k] if p in bench_parts)


def score(parts):
    from photo2fcstd import bench, recognise, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    out = {}
    for part in parts:
        rec = recs[part]
        if not rec.get("revolve"):
            continue
        sp = recognise.revolve_spec(bench.photos_of(part)[:3], rec, name=part)
        sc = SS.score_one(sp, ideal[part])
        out[part] = {"f1": round(float((sc.get("primitive_f1") or {}).get("f1", 0.0)), 4), "region_iou": round(float(sc["region_iou"]), 4),
                     "rec": {k: rec.get(k) for k in ("revolve", "openings", "face_photo_index")}}
    return out


def main():
    res = score(round_parts())
    json.dump(res, open(GOLDEN, "w"), indent=1, sort_keys=True)
    f1 = [r["f1"] for r in res.values()]
    print("recorded %d round parts: mean F1 %.3f exact %.0f%%" % (len(res), sum(f1) / len(f1), 100 * sum(x >= 0.999 for x in f1) / len(f1)))


if __name__ == "__main__":
    main()
