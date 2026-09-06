import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DUMP = os.path.join(ROOT, "tools", "dump_shape.py")


def one(part):
    import trimesh
    from photo2fcstd import bench, design, score, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    photos = bench.photos_of(part)[:3]
    d = tempfile.mkdtemp(prefix="sb_")
    out = os.path.join(d, part + ".FCStd")
    try:
        rec = recs[part]
        res = design.design(photos, out, rec=rec)
        if res.get("model", 1) is None:
            return {"part": part, "refused": res["reason"]}
        subprocess.run([os.path.expanduser(os.environ.get("FREECADCMD", "~/Code/FreeCAD/build/release/bin/FreeCADCmd")), DUMP], capture_output=True, timeout=300,
                       env={**os.environ, "FC_IN": out, "FC_MESH": out + ".npz", "FC_INFO": out + ".json"})
        m = np.load(out + ".npz"); model = trimesh.Trimesh(m["V"], m["T"])
        iou3d, _ = score.best_iou(trimesh.load(bench.truth_of(part)), model)
        sp = json.load(open(os.path.splitext(out)[0] + ".spec.json"))
        sc = SS.score_one(sp, ideal[part])
        return {"part": part, "tier": res.get("tier"), "mode": sp.get("mode"), "iou3d": round(float(iou3d), 4), "region": round(float(sc["region_iou"]), 4),
                "f1": round(float((sc.get("primitive_f1") or {}).get("f1", 0.0)), 4), "verify": res.get("verify", {}).get("silhouette_iou"),
                "note": (sp.get("revolve") or {}).get("length_note", "")[:60]}
    except Exception as e:
        return {"part": part, "error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main(parts):
    rows = []
    with Pool(int(os.environ.get("SOLID_WORKERS", "2"))) as pool:
        for k, r in enumerate(pool.imap_unordered(one, parts), 1):
            rows.append(r)
            if k % 10 == 0 or k == len(parts):
                print("progress %d/%d" % (k, len(parts)), flush=True)
    out = os.path.join(ROOT, "runs", os.environ.get("SOLID_BENCH_OUT", "solid_bench.json"))
    json.dump(sorted(rows, key=lambda r: r["part"]), open(out, "w"), indent=1)
    ok = [r for r in rows if "iou3d" in r]; ref = [r for r in rows if "refused" in r]; err = [r for r in rows if "error" in r]
    iou = np.array([r["iou3d"] for r in ok])
    print("SOLID BENCH n=%d built=%d refused=%d errors=%d | 3D IoU mean %.3f median %.3f | >=0.8: %d  >=0.6: %d  <0.3: %d | region %.3f f1 %.3f"
          % (len(rows), len(ok), len(ref), len(err), iou.mean(), np.median(iou), (iou >= 0.8).sum(), (iou >= 0.6).sum(), (iou < 0.3).sum(), np.mean([r["region"] for r in ok]), np.mean([r["f1"] for r in ok])))
    for m in sorted(set(r.get("mode") for r in ok)):
        rs = [r for r in ok if r.get("mode") == m]
        print("  mode %-8s n=%3d  3D IoU %.3f  >=0.8 %d" % (m, len(rs), np.mean([r["iou3d"] for r in rs]), sum(r["iou3d"] >= 0.8 for r in rs)))
    if err:
        print("  errors:", list(dict.fromkeys(r["error"] for r in err))[:3])


if __name__ == "__main__":
    main(sys.argv[1:] or open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split())
