import json
import os
import sys

import FreeCAD
import Part
from FreeCAD import Vector


def plane_basis(face):
    n = face.Surface.Axis
    u = Vector(1, 0, 0) if abs(n.dot(Vector(1, 0, 0))) < 0.9 else Vector(0, 1, 0)
    u = (u - n * u.dot(n)).normalize()
    return face.Surface.Position, u, n.cross(u)


def to_2d(pt, origin, u, v):
    d = pt - origin
    return [round(d.dot(u), 4), round(d.dot(v), 4)]


def edge_points(e, basis, per_edge=24):
    try:
        pts = e.discretize(Number=max(per_edge, 2))
    except Exception:
        pts = [w.Point for w in e.Vertexes]
    return [to_2d(p, *basis) for p in pts]


def edge_primitive(e, n, basis=None):
    c = e.Curve
    if isinstance(c, Part.Line) or isinstance(c, Part.LineSegment):
        out = {"type": "line", "p0": [round(v, 3) for v in e.Vertexes[0].Point], "p1": [round(v, 3) for v in e.Vertexes[-1].Point]}
    elif isinstance(c, Part.Circle):
        full = abs(e.Length - 2 * 3.141592653589793 * c.Radius) < 1e-4
        out = {"type": "circle" if full else "arc", "c": [round(v, 3) for v in c.Center], "r": round(c.Radius, 4), "span_deg": round(e.Length / c.Radius * 57.29578, 1)}
    else:
        out = {"type": c.__class__.__name__.lower(), "len": round(e.Length, 3)}
    if basis is not None:
        out["xy"] = edge_points(e, basis)
        if out["type"] in ("arc", "circle"):
            out["c2"] = to_2d(c.Center, *basis)
            out["r2"] = round(c.Radius, 4)
    return out


def largest_planar_face(shape):
    faces = [f for f in shape.Faces if isinstance(f.Surface, Part.Plane)]
    return max(faces, key=lambda f: f.Area) if faces else None


def candidate_faces(solid):
    groups = {}
    for f in solid.Faces:
        if not isinstance(f.Surface, Part.Plane):
            continue
        n = f.Surface.Axis
        key = tuple(round(abs(v), 2) for v in (n.x, n.y, n.z))
        if f.Area > groups.get(key, (0.0, None))[0]:
            groups[key] = (f.Area, f)
    return [f for _, f in groups.values()]


def prism_fit(solid, face):
    n = face.Surface.Axis
    zs = [v.Point.dot(n) for v in solid.Vertexes]
    depth = max(zs) - min(zs)
    ratio = (face.Area * depth / solid.Volume) if solid.Volume else 1e9
    return depth, ratio


def base_face(solid):
    cands = candidate_faces(solid)
    if not cands:
        return None, 0.0, 0.0
    scored = [(abs(prism_fit(solid, f)[1] - 1.0), f) for f in cands]
    _, face = min(scored, key=lambda t: (t[0], -t[1].Area))
    depth, ratio = prism_fit(solid, face)
    return face, depth, ratio


def describe(path):
    shape = Part.read(path)
    solids = shape.Solids or [shape]
    solid = max(solids, key=lambda s: s.Volume)
    face, depth, ratio = base_face(solid)
    if face is None:
        return {"prism": False, "reason": "no planar face"}
    n = face.Surface.Axis
    prism = abs(ratio - 1.0) < 0.02
    basis = plane_basis(face)
    loops = [[edge_primitive(e, n, basis) for e in w.Edges] for w in [face.OuterWire] + [w for w in face.Wires if not w.isSame(face.OuterWire)]]
    counts = {}
    for lp in loops:
        for e in lp:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    return {"prism": bool(prism), "fit": round(ratio, 4), "depth": round(depth, 3), "face_area": round(face.Area, 3), "volume": round(solid.Volume, 3),
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
