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
sys.path.insert(0, os.path.join(ROOT, "tools"))


def one(part):
    import trimesh
    import board_bench as B
    import test_board_path as T
    from photo2fcstd import axis_model, bench, carve, make_target, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    d = tempfile.mkdtemp(prefix="ax_")
    try:
        m = B.rest_on_largest_face(trimesh.load(bench.truth_of(part)))
        m.apply_scale(B.PART_MM / float(np.max(m.extents[:2])))
        if m.extents[2] < 2 * B.VOXEL:
            return None
        W, Hh = make_target.COLS * make_target.SQUARE_MM, make_target.ROWS * make_target.SQUARE_MM
        centre = np.array([W / 2, Hh / 2, 0.0])
        m.apply_translation(centre - np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        K = T._camera(B.SIZE[0], B.SIZE[1], B.FOCAL)
        photos = []
        for k, (az, el) in enumerate(B.VIEWS):
            eye = centre + 520.0 * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
            R, t = T._look_at(eye, target=centre)
            path = os.path.join(d, "v%d.jpg" % k)
            cv2.imwrite(path, cv2.cvtColor(B.render_mesh(m, R, t, K), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            photos.append(path)
        carved = carve.from_photos(photos, voxel_mm=B.VOXEL)
        scores = []
        for ax in (0, 1, 2):
            try:
                spec = carve.spec_from_carve(carved, part, axis=ax, views=carved["board_views"], masks=carved["board_masks"])
                scores.append(float(SS.score_one(spec, ideal[part])["region_iou"]))
            except Exception:
                scores.append(0.0)
        if max(scores) <= 0:
            return None
        return {"part": part, "x": axis_model.features(carved).tolist(), "iou": scores, "best": int(np.argmax(scores))}
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main():
    from photo2fcstd import sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p]) and os.path.exists(os.path.join(ROOT, "data", "printcad", "PrintCAD", "stl_from_step", p + ".stl"))]
    with Pool(int(os.environ.get("BOARD_WORKERS", "24"))) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    out = os.path.join(ROOT, "data", "axis_rows_board.json")
    json.dump(rows, open(out, "w"))
    print("%d of %d parts labelled -> %s ; best-axis distribution %s" % (len(rows), len(parts), out, {a: sum(r["best"] == a for r in rows) for a in (0, 1, 2)}))


if __name__ == "__main__":
    main()
