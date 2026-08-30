import json
import os
import sys

import FreeCAD
import Part
from FreeCAD import Vector


def edge_primitive(e, n):
    c = e.Curve
    if isinstance(c, Part.Line) or isinstance(c, Part.LineSegment):
        return {"type": "line", "p0": [round(v, 3) for v in e.Vertexes[0].Point], "p1": [round(v, 3) for v in e.Vertexes[-1].Point]}
    if isinstance(c, Part.Circle):
        full = abs(e.Length - 2 * 3.141592653589793 * c.Radius) < 1e-4
        return {"type": "circle" if full else "arc", "c": [round(v, 3) for v in c.Center], "r": round(c.Radius, 4), "span_deg": round(e.Length / c.Radius * 57.29578, 1)}
    return {"type": c.__class__.__name__.lower(), "len": round(e.Length, 3)}


def largest_planar_face(shape):
    faces = [f for f in shape.Faces if isinstance(f.Surface, Part.Plane)]
    return max(faces, key=lambda f: f.Area) if faces else None


def describe(path):
    shape = Part.read(path)
    solids = shape.Solids or [shape]
    solid = max(solids, key=lambda s: s.Volume)
    face = largest_planar_face(solid)
    if face is None:
        return {"prism": False, "reason": "no planar face"}
    n = face.Surface.Axis
    zs = [v.Point.dot(n) for v in solid.Vertexes]
    depth = max(zs) - min(zs)
    prism = abs(solid.Volume - face.Area * depth) < 0.02 * solid.Volume
    loops = [[edge_primitive(e, n) for e in w.Edges] for w in [face.OuterWire] + [w for w in face.Wires if not w.isSame(face.OuterWire)]]
    counts = {}
    for lp in loops:
        for e in lp:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    return {"prism": bool(prism), "depth": round(depth, 3), "face_area": round(face.Area, 3), "volume": round(solid.Volume, 3),
            "n_loops": len(loops), "counts": counts, "loops": loops}


def main():
    ids = open(os.environ["IDS"]).read().split()
    root = os.environ["STEP_DIR"]
    out = {}
    for k in ids:
        try:
            out[k] = describe(os.path.join(root, k + ".stp"))
        except Exception as exc:
            out[k] = {"prism": False, "reason": str(exc)[:120]}
    json.dump(out, open(os.environ["OUT"], "w"), indent=0)
    prisms = sum(1 for v in out.values() if v.get("prism"))
    types = {}
    for v in out.values():
        for t, c in v.get("counts", {}).items():
            types[t] = types.get(t, 0) + c
    print("REPORT %d parts, %d are extrusions of their largest planar face (%.0f%%); edge types %s" % (len(out), prisms, 100.0 * prisms / max(len(out), 1), types))


main()
