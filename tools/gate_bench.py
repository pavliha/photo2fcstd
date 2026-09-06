import json
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
THRESHOLD = 0.6


def one(part):
    from photo2fcstd import bench, cli, design, recognise, sketch_score as SS
    photos = bench.photos_of(part)[:3]
    d = tempfile.mkdtemp(prefix="gb_")
    out = os.path.join(d, part + ".FCStd")
    try:
        rec = {**recognise._recognise_program_live(photos), "single_extrusion": True}
        spec = recognise.revolve_spec(photos, rec, name=part) if rec.get("revolve") \
            else recognise.route(photos, name=part, rec=rec)[0]
        tilt = {"applied": False}
        sp = os.path.join(d, part + ".spec.json")
        json.dump(spec, open(sp, "w"))
        r = cli.freecad_build(sp, out)
        if os.environ.get("P2F_FACEPOSE", "1") == "1" and r["valid"] and spec.get("mode") == "plan":
            from photo2fcstd import facepose
            raw = max((design.verify(out, photos, {"face_photo_index": i})["silhouette_iou"] or 0.0) for i in range(len(photos)))
            if raw < facepose.RAW_VERIFY_MAX:
                spec2, tilt = facepose.maybe_rectify(photos, spec, name=part)
                if tilt.get("applied"):
                    spec = spec2; json.dump(spec, open(sp, "w")); r = cli.freecad_build(sp, out)
        sc = SS.score_one(spec, IDEAL[part])
        f1 = float((sc.get("primitive_f1") or {}).get("f1", 0.0)); region = float(sc["region_iou"])
        ious = []
        if r["valid"]:
            if tilt.get("applied"):
                v = design.verify(out, photos, {"face_photo_index": 0}, mask=tilt["mask"])
                ious.append(v["silhouette_iou"] if v["silhouette_iou"] is not None else 0.0)
            else:
                for i in range(len(photos)):
                    v = design.verify(out, photos, {"face_photo_index": i})
                    ious.append(v["silhouette_iou"] if v["silhouette_iou"] is not None else 0.0)
        iou = max(ious) if ious else None
        refused = (not r["valid"]) or iou is None or iou < THRESHOLD
        return {"part": part, "mode": spec.get("mode"), "valid": bool(r["valid"]), "f1": f1, "region_iou": region,
                "iou": iou, "refused": bool(refused), "rectified": bool(tilt.get("applied")),
                "tilt": float(tilt["tilt"]) if tilt.get("applied") else None,
                "tilts": tilt.get("tilts"), "agreement": tilt.get("agreement"), "view": tilt.get("view"),
                "face_inliers": tilt.get("face_inliers")}
    except Exception as e:
        return {"part": part, "error": str(e)[:200]}


def main(n=150, seed=0):
    from photo2fcstd import bench, sketch_score as SS
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0]
    rng = np.random.default_rng(seed)
    parts = [parts[i] for i in rng.choice(len(parts), min(n, len(parts)), replace=False)]
    with Pool(int(os.environ.get("GATE_WORKERS", "6"))) as pool:
        rows = pool.map(one, parts)
    out = os.path.join(ROOT, "runs", os.environ.get("GATE_BENCH_OUT", "gate_bench.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    valid = [r for r in ok if r["valid"]]
    acc = [r for r in ok if not r["refused"]]
    ref = [r for r in ok if r["refused"]]
    good = [r for r in ok if r["f1"] >= 0.6]
    false_ref = [r for r in good if r["refused"]]
    with_iou = [r for r in valid if r["iou"] is not None]
    rho = None
    if len(with_iou) > 5:
        from scipy.stats import spearmanr
        rho = float(spearmanr([r["iou"] for r in with_iou], [r["f1"] for r in with_iou]).correlation)
    rect = [r for r in ok if r.get("rectified")]
    print("RECTIFIED %d parts; mean region IoU all %.3f" % (len(rect), float(np.mean([r["region_iou"] for r in ok])) if ok else -1))
    print("GATEBENCH n=%d errors=%d valid=%d accepted=%d refused=%d | F1 accepted %.3f refused %.3f | "
          "good(F1>=0.6)=%d false_refused=%d (%.1f%%) | spearman(iou,f1)=%s"
          % (len(rows), len(rows) - len(ok), len(valid), len(acc), len(ref),
             float(np.mean([r["f1"] for r in acc])) if acc else -1, float(np.mean([r["f1"] for r in ref])) if ref else -1,
             len(good), len(false_ref), 100.0 * len(false_ref) / max(len(good), 1), rho))
    for m in sorted(set(r.get("mode") for r in ok)):
        rs = [r for r in ok if r.get("mode") == m]
        print("  mode %-8s n=%d refused=%d meanIoU=%s" % (m, len(rs), sum(r["refused"] for r in rs),
              round(float(np.mean([r["iou"] for r in rs if r["iou"] is not None])), 3) if any(r["iou"] is not None for r in rs) else None))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
