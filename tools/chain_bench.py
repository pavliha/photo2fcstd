import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("P2F_SQUARE_MM", "30")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def one(part):
    import cv2, trimesh
    import board_bench as B
    import test_board_path as T
    from photo2fcstd import bench, design, make_target, score
    recs = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    d = tempfile.mkdtemp(prefix="ch_")
    try:
        m = B.rest_on_largest_face(trimesh.load(bench.truth_of(part)))
        m.apply_scale(B.PART_MM / float(np.max(m.extents[:2])))
        if m.extents[2] < 2 * B.VOXEL:
            return {"part": part, "error": "too thin"}
        W, Hh = make_target.COLS * make_target.SQUARE_MM, make_target.ROWS * make_target.SQUARE_MM
        centre = np.array([W / 2, Hh / 2, 0.0])
        m.apply_translation(centre - np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        K = T._camera(B.SIZE[0], B.SIZE[1], B.FOCAL); photos = []
        for k, (az, el) in enumerate(B.VIEWS):
            eye = centre + 520.0 * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
            R, t = T._look_at(eye, target=centre); p = os.path.join(d, "v%d.jpg" % k)
            cv2.imwrite(p, cv2.cvtColor(B.render_mesh(m, R, t, K), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); photos.append(p)
        out = os.path.join(d, part + ".FCStd")
        res = design.design(photos, out, rec={**recs.get(part, {}), "single_extrusion": True})
        if res.get("model", 1) is None:
            return {"part": part, "refused": res["reason"]}
        subprocess.run([os.path.expanduser(os.environ.get("FREECADCMD", "~/Code/FreeCAD/build/release/bin/FreeCADCmd")), os.path.join(ROOT, "tools", "dump_shape.py")],
                       capture_output=True, timeout=300, env={**os.environ, "FC_IN": out, "FC_MESH": out + ".npz", "FC_INFO": out + ".json"})
        mm = np.load(out + ".npz"); model = trimesh.Trimesh(mm["V"], mm["T"])
        iou, _ = score.best_iou(m, model)
        return {"part": part, "tier": res.get("tier"), "fitter": res.get("fitter", "tracer"), "iou3d": round(float(iou), 4), "verify": res.get("verify", {}).get("silhouette_iou")}
    except Exception as e:
        return {"part": part, "error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main(parts):
    rows = []
    for k, part in enumerate(parts, 1):
        rows.append(one(part))
        print("progress %d/%d %s %s" % (k, len(parts), part, rows[-1].get("iou3d", rows[-1].get("refused", rows[-1].get("error", "")))[:40] if isinstance(rows[-1].get("iou3d", rows[-1].get("refused", rows[-1].get("error", ""))), str) else rows[-1].get("iou3d")), flush=True)
        json.dump(rows, open(os.path.join(ROOT, "runs", os.environ.get("CHAIN_OUT", "chain_bench.json")), "w"), indent=1)
    ok = [r for r in rows if "iou3d" in r]; iou = np.array([r["iou3d"] for r in ok])
    print("CHAIN n=%d built=%d refused=%d errors=%d | 3D IoU mean %.3f median %.3f >=0.8 %d <0.3 %d" % (len(rows), len(ok), sum("refused" in r for r in rows), sum("error" in r for r in rows), iou.mean() if len(iou) else 0, np.median(iou) if len(iou) else 0, (iou >= 0.8).sum(), (iou < 0.3).sum()))
    for f in sorted(set(r["fitter"] for r in ok)):
        rs = [r for r in ok if r["fitter"] == f]; print("  %-45s n=%d  3D IoU %.3f  >=0.8 %d" % (f[:45], len(rs), np.mean([r["iou3d"] for r in rs]), sum(r["iou3d"] >= 0.8 for r in rs)))


if __name__ == "__main__":
    main(sys.argv[1:] or open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split())
