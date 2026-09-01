"""How much of a sketch's internal structure does a visual hull actually recover?

The drawings carving produces are outlines: on T-LESS connector housings it returns a smooth
profile where the truth has slots and steps. A visual hull can only see a cavity that breaks the
silhouette from some direction, so a through-hole is recoverable and a blind pocket is not. Region
IoU hides this because internal loops carry little area.
"""
import json, os, sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from photo2fcstd import axis_model, carve as C, sketch_score as SS, tless  # noqa: E402
from tless_sketch import cross_section  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
PERM = {0: [1, 2, 0], 1: [2, 0, 1], 2: [0, 1, 2]}


def loop_areas(rings):
    out = []
    for r in rings:
        try:
            p = Polygon(np.asarray(r, float))
            if p.is_valid and p.area > 0:
                out.append(p.area)
        except Exception:
            continue
    return sorted(out, reverse=True)


def main():
    ref = json.load(open(os.path.join(ROOT, "data", "tless_sketch.json")))
    rows = []
    for r in ref:
        try:
            mesh = trimesh.load(tless.model_path(r["obj"]))
            truth = cross_section(mesh, r["axis"])
            if not truth:
                continue
            carved, _ = tless.carve_object(r["obj"], every=90, voxel_mm=1.0)
            if carved is None:
                continue
            cc = dict(carved)
            cc["points_mm"] = carved["points_mm"][:, PERM[r["axis"]]]
            spec = C.spec_from_carve(cc, name="o", axis=2)
            got = SS.spec_rings(spec)
            ta, ga = loop_areas(truth), loop_areas(got)
            if not ta or not ga:
                continue
            rows.append({"obj": r["obj"], "truth_loops": len(ta), "got_loops": len(ga),
                         "truth_inner_frac": float(sum(ta[1:]) / ta[0]) if len(ta) > 1 else 0.0,
                         "got_inner_frac": float(sum(ga[1:]) / ga[0]) if len(ga) > 1 else 0.0})
        except Exception:
            continue
    json.dump(rows, open(os.path.join(ROOT, "data", "loop_recovery.json"), "w"))
    if not rows:
        print("nothing measurable")
        return
    tl = np.array([r["truth_loops"] for r in rows])
    gl = np.array([r["got_loops"] for r in rows])
    with_inner = tl > 1
    print("T-LESS, n=%d objects, axis forced to the reference\n" % len(rows))
    print("  %-40s %8.2f" % ("loops in the real cross-section", tl.mean()))
    print("  %-40s %8.2f" % ("loops the carve recovers", gl.mean()))
    print("  %-40s %7.0f%%" % ("objects whose truth has internal loops", 100 * with_inner.mean()))
    if with_inner.any():
        rec = np.mean([r["got_loops"] > 1 for r, w in zip(rows, with_inner) if w])
        print("  %-40s %7.0f%%" % ("  ... of those, we recover any", 100 * rec))
        print("  %-40s %8.3f" % ("  internal area, truth (fraction of outer)",
                                 np.mean([r["truth_inner_frac"] for r, w in zip(rows, with_inner) if w])))
        print("  %-40s %8.3f" % ("  internal area, recovered",
                                 np.mean([r["got_inner_frac"] for r, w in zip(rows, with_inner) if w])))
    print("\n  a loop carrying 2%% of the outer area moves region IoU by about 0.02,")
    print("  which is why an outline with no internal structure still scores 0.5 to 0.86")


if __name__ == "__main__":
    main()
