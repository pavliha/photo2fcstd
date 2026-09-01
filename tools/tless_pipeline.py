"""Run the shipped photo-to-sketch pipeline on a second real photo domain.

Every number in PLAN.md comes from PrintCAD: one set of tables, one phone, one lighting.
T-LESS is thirty industrial parts photographed on a calibrated turntable, with CAD meshes.
The reference is the mesh's own cross-section perpendicular to its extrusion axis, which is
the sketch the part was drawn from.
"""
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from photo2fcstd import analysis, sketch_score as SS, spec as spec_mod, stats, tless
from tless_sketch import cross_section

OUT = "data/tless_pipeline.json"


def record_from_rings(rings):
    """Wrap the cross-section rings as the sketch record score_one expects."""
    def ring(points):
        pts = [[float(x), float(y)] for x, y in points]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return [{"type": "line", "p0": pts[i], "p1": pts[(i + 1) % len(pts)],
                 "xy": [pts[i], pts[(i + 1) % len(pts)]]} for i in range(len(pts))]
    loops = [ring(r) for r in rings if len(r) >= 4]
    if not loops:
        return None
    outer = np.asarray(rings[0], float)
    span = float(max(np.ptp(outer[:, 0]), np.ptp(outer[:, 1])))
    return {"loops": loops, "n_loops": len(loops),
            "counts": {"line": sum(len(l) for l in loops)},
            "depth": span * 0.5, "prism": True, "fit": 1.0}


def photos_for(obj_id, count=3):
    scene = tless.scene_dir(obj_id)
    rgb = os.path.join(scene, "rgb")
    if not os.path.isdir(rgb):
        return []
    names = sorted(os.listdir(rgb))
    if len(names) < count:
        return []
    step = max(1, len(names) // count)
    return [os.path.join(rgb, names[i * step]) for i in range(count)]


def main():
    axes = {int(r["obj"]): int(r["axis"]) for r in json.load(open("data/tless_sketch.json"))}
    rows = []
    for obj_id in sorted(axes):
        photos = photos_for(obj_id)
        if len(photos) < 3:
            continue
        try:
            mesh = trimesh.load(tless.model_path(obj_id))
            rings = cross_section(mesh, axes[obj_id])
            if rings is None:
                continue
            reference = record_from_rings(rings)
            if reference is None:
                continue
            views = [analysis.view(p) for p in photos]
            doc = spec_mod.assemble(views, name="obj_%02d" % obj_id, log=lambda *a: None)
            got = SS.score_one(doc, reference)
        except Exception as exc:
            print("  obj %02d failed: %s" % (obj_id, exc), file=sys.stderr)
            continue
        f1 = got.get("primitive_f1") or {}
        rows.append({"obj": obj_id, "region_iou": got.get("region_iou"),
                     "f1": f1.get("f1"), "drawn": f1.get("drawn"), "wanted": f1.get("wanted")})
    json.dump(rows, open(OUT, "w"))
    good = [r for r in rows if r["region_iou"] is not None]
    print("T-LESS: %d objects scored" % len(good))
    if good:
        m, lo, hi = stats.mean_ci([r["region_iou"] for r in good])
        print("  region IoU   %.3f [%.3f, %.3f]   (PrintCAD test set 0.629)" % (m, lo, hi))
        f = [r["f1"] for r in good if r["f1"] is not None]
        if f:
            m2, lo2, hi2 = stats.mean_ci(f)
            print("  primitive F1 %.3f [%.3f, %.3f]   (PrintCAD test set 0.612)" % (m2, lo2, hi2))


if __name__ == "__main__":
    main()
