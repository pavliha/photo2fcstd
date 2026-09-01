"""Carve to a sketch from real photographs, and from renders of the same object at the same poses.

Every carve-to-sketch number so far comes from rendered silhouettes. T-LESS has real photographs
with dataset poses and real segmentation, and a CAD mesh to score against, so the gap between the
two arms is what real capture costs measured rather than modelled. The reference is the mesh's own
cross-section perpendicular to the extrusion axis, which is the sketch the part was drawn from.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from photo2fcstd import axis_model, carve as C, carve_check as CC, sketch_score as SS, tless  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
OBJECTS = list(range(1, 31))
EVERY = 60


def cross_section(mesh, axis):
    """The polygon the part would have been drawn as, sliced halfway up its extrusion axis."""
    n = np.zeros(3)
    n[axis] = 1.0
    origin = mesh.bounds.mean(axis=0)
    try:
        sec = mesh.section(plane_origin=origin, plane_normal=n)
        if sec is None:
            return None
        planar, _ = sec.to_planar()
    except Exception:
        return None
    polys = [p for p in planar.polygons_full] if hasattr(planar, "polygons_full") else []
    if not polys:
        return None
    best = max(polys, key=lambda p: p.area)
    rings = [np.asarray(best.exterior.coords, float)]
    rings += [np.asarray(i.coords, float) for i in best.interiors]
    return rings


def section_variation(mesh, axis):
    """How much the cross-section area changes along an axis. Low means one sketch describes it."""
    n = np.zeros(3)
    n[axis] = 1.0
    lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
    areas = []
    for t in (0.3, 0.5, 0.7):
        o = mesh.bounds.mean(axis=0).copy()
        o[axis] = lo + t * (hi - lo)
        try:
            planar, _ = mesh.section(plane_origin=o, plane_normal=n).to_planar()
            areas.append(max(p.area for p in planar.polygons_full))
        except Exception:
            return 1.0
    a = np.array(areas, float)
    return float((a.max() - a.min()) / a.max()) if a.max() > 0 else 1.0


def reference_axis(mesh, tol=0.20):
    """The axis a single sketch can describe, or None. T-LESS parts are not prisms like
    PrintCAD's - their section varies 0.09 to 0.78 depending on direction - so the reference
    has to be the direction that is closest to constant, not whichever one we happen to pick."""
    v = [section_variation(mesh, a) for a in (0, 1, 2)]
    best = int(np.argmin(v))
    return (best, v[best]) if v[best] < tol else (None, min(v))


def sketch_of(carved, name, axis=None):
    ax = int(np.argmax(MODEL.predict_proba(axis_model.features(carved))[:, 1])) if axis is None else axis
    perm = {0: [1, 2, 0], 1: [2, 0, 1], 2: [0, 1, 2]}[ax]
    cc = dict(carved)
    cc["points_mm"] = carved["points_mm"][:, perm]
    return C.spec_from_carve(cc, name=name, axis=2), ax


def one(obj_id):
    try:
        mesh = trimesh.load(tless.model_path(obj_id))
        ax, var = reference_axis(mesh)
        if ax is None:
            return None
        real, _ = tless.carve_object(obj_id, every=EVERY, voxel_mm=0.8)
        if real is None or len(real["points_mm"]) < 200:
            return None
        chosen = int(np.argmax(MODEL.predict_proba(axis_model.features(real))[:, 1]))
        spec_r, _ = sketch_of(real, "obj%02d" % obj_id, axis=ax)
        rings = cross_section(mesh, ax)
        if not rings:
            return None
        record = {"loops": [[{"xy": r.tolist(), "type": "line"}] for r in rings],
                  "n_loops": len(rings), "prism": True, "counts": {}}
        pairs = tless.views_of(obj_id, every=EVERY)
        views = [v for v, _ in pairs] if pairs and isinstance(pairs[0], tuple) else pairs
        try:
            masks = [CC.silhouette(mesh, v) for v in views]
            synth = C.carve(views, masks, voxel_mm=0.8)
            spec_s, _ = sketch_of(synth, "obj%02d" % obj_id, axis=ax) if synth is not None else (None, None)
        except Exception:
            spec_s = None
        out = {"obj": obj_id, "axis": ax, "variation": var, "axis_model_agreed": chosen == ax,
               "real": SS.score_one(spec_r, record)["region_iou"],
               "trivial": SS.trivial_score(record)}
        if spec_s is not None:
            out["synth"] = SS.score_one(spec_s, record)["region_iou"]
        return out
    except Exception:
        return None


def main():
    with Pool(4) as pool:
        rows = [r for r in pool.map(one, OBJECTS) if r]
    json.dump(rows, open(os.path.join(ROOT, "data", "tless_sketch.json"), "w"))
    if not rows:
        print("no prismatic T-LESS objects produced a sketch")
        return
    keen = [r for r in rows if 1 - r["trivial"] >= 0.15]
    both = [r for r in keen if "synth" in r]
    print("%d of 30 T-LESS objects have an axis one sketch can describe, %d discriminating" % (len(rows), len(keen)))
    print("the axis model picked that same axis on %.0f%% of them\n"
          % (100 * np.mean([r["axis_model_agreed"] for r in rows])))
    print("  %-34s %8s" % ("", "sketch IoU"))
    print("  %-34s %8.3f" % ("a circle, ignoring the photos", np.mean([r["trivial"] for r in keen])))
    print("  %-34s %8.3f" % ("carved from real photographs", np.mean([r["real"] for r in keen])))
    if both:
        print("  %-34s %8.3f" % ("carved from renders, same poses", np.mean([r["synth"] for r in both])))
        print("  %-34s %8.3f" % ("  (real, on those same objects)", np.mean([r["real"] for r in both])))
        print("\n  what real capture costs: %+.3f over n=%d"
              % (np.mean([r["real"] - r["synth"] for r in both]), len(both)))


if __name__ == "__main__":
    main()
