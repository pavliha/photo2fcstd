"""Read the Fusion 360 Gallery reconstruction dataset as ideal sketches.

PrintCAD's ground truth has to be *inferred*: the reference face is guessed from a STEP file and
`sketch_score.trustworthy()` throws out about 45% of it, which is why every number in this
repository is quoted on ~1,047 parts. Fusion 360 Gallery carries the sketch a person actually drew,
as `SketchLine` / `SketchArc` / `SketchCircle` with coordinates, plus the extrude distance - so
nothing is inferred and the prism and area-times-depth tests stop being needed.

Licence: non-commercial research only, and the dataset may not be redistributed whole. Everything
this writes stays out of the repository.
"""
import glob, json, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.expanduser(os.environ.get("P2F_FUSION360", "~/.cache/photo2fcstd/fusion360/r1.0.1/reconstruction"))
STEPS = 24


def _p(point):
    """Profile-curve points are in the sketch's own 2D frame, not the world.

    The `transform` on a sketch maps that frame into the world, and two thirds of these sketches sit
    on a non-XY plane - but the curve coordinates are already local, with z zero. Projecting them
    through the transform collapses the sketch to a line: design 100243_9fb796fe_0005 carries a
    y_axis of (0, 0, -1) while its own points vary in x and y. About 6% of designs do carry a
    non-zero z, and those are ambiguous, so `record_from` drops them rather than guessing.
    """
    return float(point.get("x", 0.0)), float(point.get("y", 0.0))


def flat(curve, tol=1e-6):
    return all(abs(float(curve[k].get("z", 0.0))) <= tol
               for k in ("start_point", "end_point", "center_point") if k in curve)


def sample(curve):
    kind = curve.get("type")
    if kind == "Line3D":
        return [_p(curve["start_point"]), _p(curve["end_point"])]
    if kind == "Circle3D":
        cx, cy = _p(curve["center_point"])
        r = float(curve["radius"])
        return [(cx + r * math.cos(t), cy + r * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 4 * STEPS)]
    if kind == "Arc3D":
        cx, cy = _p(curve["center_point"])
        sx, sy = _p(curve["start_point"])
        ex, ey = _p(curve["end_point"])
        r = float(curve["radius"])
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(ey - cy, ex - cx)
        want = abs(float(curve.get("end_angle", 0.0)) - float(curve.get("start_angle", 0.0)))
        ccw = (a1 - a0) % (2 * math.pi)
        cw = (a0 - a1) % (2 * math.pi)
        span = ccw if abs(ccw - want) <= abs(cw - want) else -cw
        return [(cx + r * math.cos(a0 + span * f), cy + r * math.sin(a0 + span * f))
                for f in np.linspace(0, 1, STEPS)]
    return []


KINDS = {"Line3D": "line", "Arc3D": "arc", "Circle3D": "circle"}


def record_from(design):
    """One ideal-sketch record: the first sketch that is extruded, and by how much."""
    entities = design.get("entities", {})
    timeline = design.get("timeline", [])
    extrudes = [entities[e["entity"]] for e in timeline
                if entities.get(e["entity"], {}).get("type") == "ExtrudeFeature"]
    if not extrudes:
        return None
    ex = extrudes[0]
    depth = abs(float(((ex.get("extent_one") or {}).get("distance") or {}).get("value", 0.0)))
    wanted = {p.get("sketch") for p in ex.get("profiles", [])}
    profile_ids = {p.get("profile") for p in ex.get("profiles", [])}
    loops, counts = [], {}
    for sid in wanted:
        sketch = entities.get(sid)
        if not sketch:
            continue
        for pid, profile in (sketch.get("profiles") or {}).items():
            if pid not in profile_ids:
                continue
            for loop in profile.get("loops", []):
                els = []
                for curve in loop.get("profile_curves", []):
                    if not flat(curve):
                        return None
                    xy = sample(curve)
                    if len(xy) < 2:
                        continue
                    kind = KINDS.get(curve.get("type"))
                    if kind is None:
                        return None
                    counts[kind] = counts.get(kind, 0) + 1
                    els.append({"type": kind, "xy": [list(map(float, q)) for q in xy]})
                if els:
                    loops.append(els)
    if not loops or depth <= 0:
        return None
    area = _area(loops)
    return {"counts": counts, "loops": loops, "n_loops": len(loops), "depth": depth,
            "face_area": area, "volume": area * depth, "prism": True, "fit": 1.0,
            "source": "fusion360"}


def _area(loops):
    from photo2fcstd.sketch_score import chain_edges, polygon_of
    rings = [chain_edges([e["xy"] for e in loop]) for loop in loops]
    poly = polygon_of(rings)
    return float(poly.area) if poly is not None else 0.0


def build(limit=None, out="data/fusion360_ideal_sketches.json", tol=0.2):
    """Every record is checked against its own mesh before it is kept.

    PrintCAD's `trustworthy()` has to *guess* whether the reference face is the extrusion base, from
    a STEP file, and throws out 45% on heuristics. Here the sketch is the one a person drew and the
    dataset ships the resulting solid, so the check is direct: does the sketch's area times the
    extrude distance equal the mesh's volume. That is a measurement, not a heuristic.

    Designs with more than one extrude are skipped: this pipeline models a single extrusion, and a
    five-operation timeline has no single base face to compare against.
    """
    import trimesh
    files = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    files = files[:limit] if limit else files
    rows, why = {}, {}
    def note(k):
        why[k] = why.get(k, 0) + 1
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            design = json.load(open(f))
        except Exception:
            note("unreadable json")
            continue
        if sum(1 for v in design.get("entities", {}).values()
               if v.get("type") == "ExtrudeFeature") != 1:
            note("more than one extrude")
            continue
        try:
            rec = record_from(design)
        except Exception as exc:
            note("parse: %s" % type(exc).__name__)
            continue
        if rec is None:
            note("no usable extruded sketch")
            continue
        objs = sorted(glob.glob(os.path.join(CACHE, name + "_*.obj")))
        if not objs:
            note("no mesh to check against")
            continue
        try:
            volume = float(abs(trimesh.load(objs[-1], force="mesh").volume))
        except Exception:
            note("mesh unreadable")
            continue
        if volume <= 0 or abs(rec["face_area"] * rec["depth"] / volume - 1.0) > tol:
            note("area x depth disagrees with the mesh")
            continue
        rec["volume"] = volume
        rows[name] = rec
    json.dump(rows, open(os.path.join(ROOT, out), "w"))
    print("  %d designs read, %d verified records (%.0f%%)"
          % (len(files), len(rows), 100 * len(rows) / max(len(files), 1)))
    for k, n in sorted(why.items(), key=lambda x: -x[1]):
        print("    %5d  %s" % (n, k))
    print("  wrote %s" % out)
    return rows


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else None)
