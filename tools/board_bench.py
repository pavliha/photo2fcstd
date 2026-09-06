import json
import os
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np

os.environ.setdefault("P2F_SQUARE_MM", "30")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
SIZE = (1600, 1200)
FOCAL = 1500.0
RECS = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json"))) if os.path.exists(os.path.join(ROOT, "runs", "bench_recs.json")) else {}
VIEWS = [(0, 30), (90, 30), (180, 25), (270, 35), (45, 55), (225, 15)]
PART_MM = 80.0
VOXEL = 0.4


def rest_on_largest_face(m):
    order = np.argsort(m.extents)
    axes = np.eye(3)[[order[2], order[1], order[0]]]
    T4 = np.eye(4); T4[:3, :3] = axes if np.linalg.det(axes) > 0 else axes * np.array([[1], [1], [-1]])
    m.apply_transform(T4)
    return m


def render_mesh(mesh, R, t, K):
    import test_board_path as T
    from photo2fcstd import make_target
    img = T._render((0.0, 0.0, 0.0), R, t, K, SIZE)
    V = np.asarray(mesh.vertices, float)
    cam = (V @ R.T + t)
    p = (K @ cam.T).T
    uv = (p[:, :2] / p[:, 2:3]).astype(np.int32)
    F = np.asarray(mesh.faces)
    depth = cam[F][:, :, 2].mean(axis=1)
    normals = np.asarray(mesh.face_normals)
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, uv[F[i]], tuple(int(c * shade[i]) for c in (40, 90, 200)))
    return img


def one(part):
    import trimesh
    import test_board_path as T
    from photo2fcstd import bench, carve, make_target, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    d = tempfile.mkdtemp(prefix="bb_")
    try:
        m = trimesh.load(bench.truth_of(part))
        m = rest_on_largest_face(m)
        m.apply_scale(PART_MM / float(np.max(m.extents[:2])))
        if m.extents[2] < 2 * VOXEL:
            return {"part": part, "error": "too thin: %.2f mm at %.1f mm voxels" % (m.extents[2], VOXEL)}
        W, Hh = make_target.COLS * make_target.SQUARE_MM, make_target.ROWS * make_target.SQUARE_MM
        centre = np.array([W / 2, Hh / 2, 0.0])
        m.apply_translation(centre - np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        K = T._camera(SIZE[0], SIZE[1], FOCAL)
        photos = []
        for k, (az, el) in enumerate(VIEWS):
            eye = centre + 520.0 * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
            R, t = T._look_at(eye, target=centre)
            path = os.path.join(d, "v%d.jpg" % k)
            cv2.imwrite(path, cv2.cvtColor(render_mesh(m, R, t, K), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            photos.append(path)
        carved = carve.from_photos(photos, voxel_mm=VOXEL)
        rec = RECS.get(part, {})
        per_axis = {}
        for ax in (0, 1, 2):
            try:
                sp = carve.spec_from_carve(carved, part, axis=ax, views=carved["board_views"], masks=carved["board_masks"])
                s2 = SS.score_one(sp, ideal[part])
                per_axis[ax] = {"region": float(s2["region_iou"]), "f1": float((s2.get("primitive_f1") or {}).get("f1", 0.0))}
            except Exception as e:
                per_axis[ax] = {"region": 0.0, "f1": 0.0, "error": str(e)[:60]}
        choices = {"learned": carve.base_axis(carved)}
        spec = carve.revolve_from_carve(carved, part, rec) if rec.get("revolve") else carve.spec_from_carve(carved, part, views=carved["board_views"], masks=carved["board_masks"])
        sc = SS.score_one(spec, ideal[part])
        e_true, e_carve = np.sort(m.extents)[::-1], np.sort(carved["extents_mm"])[::-1]
        return {"part": part, "views": carved["views"], "f1": float((sc.get("primitive_f1") or {}).get("f1", 0.0)), "region_iou": float(sc["region_iou"]),
                "top_mm": float(carved["top_mm"]), "true_height_mm": float(m.extents[2]),
                "ext_err_pct": [round(100 * float((c - t) / t), 1) for c, t in zip(e_carve, e_true)],
                "section_constancy": (spec.get("outline") or {}).get("section_constancy"), "mode": spec.get("mode"),
                "per_axis": per_axis, "choices": choices}
    except Exception as e:
        import traceback
        return {"part": part, "error": "%s: %s" % (type(e).__name__, str(e)[:120]), "trace": traceback.format_exc()[-600:]}
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main(parts):
    with Pool(int(os.environ.get("BOARD_WORKERS", "4"))) as pool:
        rows = pool.map(one, parts)
    out = os.path.join(ROOT, "runs", os.environ.get("BOARD_BENCH_OUT", "board_bench.json"))
    json.dump(rows, open(out, "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    print("BOARD BENCH n=%d errors=%d  F1 %.3f  exact %.0f%%  region %.3f  height err median %.1f%%  extents err median %s%%"
          % (len(rows), len(rows) - len(ok), np.mean([r["f1"] for r in ok]), 100 * np.mean([r["f1"] >= 0.999 for r in ok]), np.mean([r["region_iou"] for r in ok]),
             100 * np.median([abs(r["top_mm"] - r["true_height_mm"]) / r["true_height_mm"] for r in ok]),
             list(np.median([r["ext_err_pct"] for r in ok], axis=0).round(1))))
    errs = [r["error"] for r in rows if "error" in r]
    if errs:
        print("  errors:", list(dict.fromkeys(errs))[:3])


if __name__ == "__main__":
    parts = sys.argv[1:] or open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split()
    main(parts)
