import json
import os
import sys

import FreeCAD as App
import Part
import Sketcher
from FreeCAD import Placement, Rotation, Vector


def sheet_with(doc, params):
    sh = doc.addObject("Spreadsheet::Sheet", "params")
    for i, (name, value, note) in enumerate(params, start=1):
        sh.set("A%d" % i, name)
        sh.set("B%d" % i, str(value))
        sh.set("C%d" % i, note)
        sh.setAlias("B%d" % i, name)
    doc.recompute()
    return sh


def elevation(doc, body, name, halves, zs, rot):
    n = len(halves)
    pts = []
    for i in range(n):
        pts += [(halves[i][0], zs[i][0]), (halves[i][0], zs[i + 1][0])]
    for i in reversed(range(n)):
        pts += [(-halves[i][0], zs[i + 1][0]), (-halves[i][0], zs[i][0])]
    sk = doc.addObject("Sketcher::SketchObject", name)
    body.addObject(sk)
    sk.Placement = Placement(Vector(0, 0, 0), rot)
    m = len(pts)
    ids = [sk.addGeometry(Part.LineSegment(Vector(pts[i][0], pts[i][1], 0),
                                           Vector(pts[(i + 1) % m][0], pts[(i + 1) % m][1], 0)), False)
           for i in range(m)]
    for i in range(m):
        sk.addConstraint(Sketcher.Constraint("Coincident", ids[i], 2, ids[(i + 1) % m], 1))
    for i in range(0, m, 2):
        sk.addConstraint(Sketcher.Constraint("Vertical", ids[i]))
    for i in range(1, m, 2):
        sk.addConstraint(Sketcher.Constraint("Horizontal", ids[i]))
    dims = [("DistanceY", 0, 0.0, "z0", "0")]
    for i in range(n):
        dims.append(("DistanceX", 2 * i, halves[i][0], "r%d" % i, halves[i][1]))
        dims.append(("DistanceY", 2 * i + 1, zs[i + 1][0], "z%d" % (i + 1), zs[i + 1][1]))
    for i in range(n):
        j = 2 * n + 2 * (n - 1 - i)
        dims.append(("DistanceX", j, -halves[i][0], "l%d" % i, "-(%s)" % halves[i][1]))
        if i >= 1:
            dims.append(("DistanceY", j + 1, zs[i][0], "zl%d" % i, zs[i][1]))
    for kind, gi, val, cname, expr in dims:
        cid = sk.addConstraint(Sketcher.Constraint(kind, -1, 1, ids[gi], 1, val))
        sk.renameConstraint(cid, cname)
        sk.setExpression(".Constraints.%s" % cname, expr)
    return sk


def radial(loop, i):
    els, jn = loop["elements"], loop["joins"]
    n = len(els)
    if els[i]["type"] != "line" or jn[i] != "P" or jn[(i + 1) % n] != "P":
        return False
    a, b = els[i - 1], els[(i + 1) % n]
    return a["type"] == "arc" and b["type"] == "arc" and (b.get("centre_of") == (i - 1) % n or a.get("centre_of") == (i + 1) % n)


def arc_dims(loop, i):
    els, jn = loop["elements"], loop["joins"]
    e = els[i]
    plain = jn[i] == "" and jn[(i + 1) % len(els)] == ""
    return (not plain) and e.get("centre_of") is None, els[i - 1]["type"] != "arc"


def centre_defined(loop, i):
    e = loop["elements"][i]
    return e["type"] == "arc" and (arc_dims(loop, i)[0] or e.get("centre_of") is not None)


def one_dim(loop, i, arc_idx):
    e, p = loop["elements"][arc_idx], loop["elements"][i]["p0"]
    dx, dy = p[0] - e["cx"], p[1] - e["cy"]
    return (i, True, False) if abs(dy) > abs(dx) else (i, False, True)


def vertex_dims(loop):
    els, kinds, jn = loop["elements"], loop["kinds"], loop["joins"]
    n = len(els)
    dims = []
    for i in range(n):
        prev_k, k = kinds[i - 1], kinds[i]
        if prev_k == "A" or k == "A":
            if jn[i] == "T":
                continue
            if jn[i] == "P":
                if radial(loop, i):
                    dims.append(one_dim(loop, i, i - 1))
                continue
            arc_idx = next((j for j in ((i - 1) % n, i) if centre_defined(loop, j)), None)
            dims.append(one_dim(loop, i, arc_idx) if arc_idx is not None else (i, True, True))
            continue
        if i == 0:
            dims.append((i, True, True))
        else:
            dims.append((i, prev_k != "V", prev_k != "H"))
    return dims


def loop_dims(loop, prefix):
    if loop["type"] == "circle":
        return [(prefix + "_cx", loop["cx"]), (prefix + "_cy", loop["cy"]), (prefix + "_r", loop["r"])]
    out = []
    for i, want_x, want_y in vertex_dims(loop):
        p0 = loop["elements"][i]["p0"]
        if want_x:
            out.append((prefix + "_x%d" % i, p0[0]))
        if want_y:
            out.append((prefix + "_y%d" % i, p0[1]))
    for i, e in enumerate(loop["elements"]):
        if e["type"] == "arc":
            want_centre, want_radius = arc_dims(loop, i)
            if want_centre:
                out += [(prefix + "_cx%d" % i, e["cx"]), (prefix + "_cy%d" % i, e["cy"])]
            if want_radius:
                out.append((prefix + "_r%d" % i, e["r"]))
    return out


def bind(sk, cid, name):
    sk.renameConstraint(cid, name)
    sk.setExpression(".Constraints." + name, "params.%s * params.mm_per_px" % name)


def arc_geometry(e, mpp):
    import math
    c = Vector(e["cx"] * mpp, e["cy"] * mpp, 0)
    a0 = math.atan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"])
    a1 = math.atan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"])
    if e["ccw"]:
        while a1 <= a0:
            a1 += 2 * math.pi
        return Part.ArcOfCircle(Part.Circle(c, Vector(0, 0, 1), e["r"] * mpp), a0, a1), 1, 2
    while a0 <= a1:
        a0 += 2 * math.pi
    return Part.ArcOfCircle(Part.Circle(c, Vector(0, 0, 1), e["r"] * mpp), a1, a0), 2, 1


def add_loop(sk, loop, prefix, mpp):
    if loop["type"] == "circle":
        g = sk.addGeometry(Part.Circle(Vector(loop["cx"] * mpp, loop["cy"] * mpp, 0), Vector(0, 0, 1), loop["r"] * mpp), False)
        bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceX", -1, 1, g, 3, loop["cx"] * mpp)), prefix + "_cx")
        bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, g, 3, loop["cy"] * mpp)), prefix + "_cy")
        bind(sk, sk.addConstraint(Sketcher.Constraint("Radius", g, loop["r"] * mpp)), prefix + "_r")
        return
    els, kinds, jn = loop["elements"], loop["kinds"], loop["joins"]
    n = len(els)
    ids, ends = [], []
    for e in els:
        if e["type"] == "line":
            ids.append(sk.addGeometry(Part.LineSegment(Vector(e["p0"][0] * mpp, e["p0"][1] * mpp, 0), Vector(e["p1"][0] * mpp, e["p1"][1] * mpp, 0)), False))
            ends.append((1, 2))
        else:
            geo, s0, s1 = arc_geometry(e, mpp)
            ids.append(sk.addGeometry(geo, False))
            ends.append((s0, s1))
    for i in range(n):
        if jn[i] == "T":
            sk.addConstraint(Sketcher.Constraint("Tangent", ids[i - 1], ends[i - 1][1], ids[i], ends[i][0]))
        elif jn[i] == "P" and not radial(loop, i - 1):
            sk.addConstraint(Sketcher.Constraint("Perpendicular", ids[i - 1], ends[i - 1][1], ids[i], ends[i][0]))
        else:
            sk.addConstraint(Sketcher.Constraint("Coincident", ids[i - 1], ends[i - 1][1], ids[i], ends[i][0]))
    for i, want_x, want_y in vertex_dims(loop):
        p0 = els[i]["p0"]
        if want_x:
            bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceX", -1, 1, ids[i], ends[i][0], p0[0] * mpp)), prefix + "_x%d" % i)
        if want_y:
            bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, ids[i], ends[i][0], p0[1] * mpp)), prefix + "_y%d" % i)
    last_line = max((i for i, e in enumerate(els) if e["type"] == "line"), default=-1)
    for i, e in enumerate(els):
        if e["type"] == "line":
            tangent_neighbour = jn[i] == "T" or jn[(i + 1) % n] == "T" or i == last_line
            if kinds[i] == "H" and not tangent_neighbour:
                sk.addConstraint(Sketcher.Constraint("Horizontal", ids[i]))
            elif kinds[i] == "V" and not tangent_neighbour:
                sk.addConstraint(Sketcher.Constraint("Vertical", ids[i]))
        else:
            want_centre, want_radius = arc_dims(loop, i)
            if want_centre:
                bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceX", -1, 1, ids[i], 3, e["cx"] * mpp)), prefix + "_cx%d" % i)
                bind(sk, sk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, ids[i], 3, e["cy"] * mpp)), prefix + "_cy%d" % i)
            elif e.get("centre_of") is not None:
                sk.addConstraint(Sketcher.Constraint("Coincident", ids[i], 3, ids[e["centre_of"]], 3))
            if want_radius:
                bind(sk, sk.addConstraint(Sketcher.Constraint("Radius", ids[i], e["r"] * mpp)), prefix + "_r%d" % i)


def outline_sketch(doc, body, name, loops, mpp):
    sk = doc.addObject("Sketcher::SketchObject", name)
    body.addObject(sk)
    for j, loop in enumerate(loops):
        add_loop(sk, loop, "o" if j == 0 else "h%d" % (j - 1), mpp)
    return sk


def build_outline(spec, doc, params):
    v = spec["outline"]
    mpp = spec["mm_per_px"]
    for j, loop in enumerate(v["loops"]):
        prefix = "o" if j == 0 else "h%d" % (j - 1)
        what = "outline" if j == 0 else "hole %d" % (j - 1)
        params += [(n, round(val, 3), "%s %s (px)" % (what, n.split("_", 1)[1])) for n, val in loop_dims(loop, prefix)]
    params.append(("depth", round(v["depth_px"], 3), v["depth_note"]))
    sheet_with(doc, params)
    body = doc.addObject("PartDesign::Body", "Body")
    body.Label = spec.get("name", "part")
    sk = outline_sketch(doc, body, "sk_outline", v["loops"], mpp)
    pad = doc.addObject("PartDesign::Pad", "pad_outline")
    body.addObject(pad)
    pad.Profile = sk
    pad.Length = v["depth_px"] * mpp
    pad.setExpression("Length", "params.depth * params.mm_per_px")
    doc.recompute()
    return body

def build_revolve(spec, doc, params):
    v = spec["revolve"]
    mpp = spec["mm_per_px"]
    prof = v["profile"]
    generic = v.get("generic")
    if generic:
        m = len(prof)
        els = [{"type": "line", "p0": prof[i], "p1": prof[(i + 1) % m]} for i in range(m)]
        kinds = ["H" if abs(e["p1"][1] - e["p0"][1]) < 1e-9 else "V" if abs(e["p1"][0] - e["p0"][0]) < 1e-9 else "F" for e in els]
        ploop = {"type": "loop", "elements": els, "kinds": kinds, "joins": [""] * m}
        params += [(n, round(val, 3), "profile %s (px)" % n.split("_", 1)[1]) for n, val in loop_dims(ploop, "p")]
    else:
        params += [("R_outer", round(v["R"], 3), "outer radius (px)"), ("t_floor", round(v["t"], 3), "floor thickness (px) - guess, measure it"), ("h_rim", round(v["h"], 3), "rim height (px) - guess, measure it")]
        params += [("R_rim", round(v["rings"][0] * v["R"], 3), "rim inner radius (px)")] if v["rings"] else []
    for j, hole in enumerate(v["holes"]):
        params += [("hole%d_cx" % j, round(hole["cx"], 3), "hole %d x (px)" % j), ("hole%d_cy" % j, round(hole["cy"], 3), "hole %d y (px)" % j), ("hole%d_r" % j, round(hole["r"], 3), "hole %d radius (px)" % j)]
    sheet_with(doc, params)
    body = doc.addObject("PartDesign::Body", "Body")
    body.Label = spec.get("name", "part")
    sk = doc.addObject("Sketcher::SketchObject", "sk_profile")
    body.addObject(sk)
    sk.AttachmentSupport = [(doc.getObject("XZ_Plane") or [o for o in body.Origin.OriginFeatures if "XZ" in o.Name][0], "")]
    sk.MapMode = "FlatFace"
    if generic:
        add_loop(sk, ploop, "p", mpp)
    else:
        pts = [(x * mpp, y * mpp) for x, y in prof]
        n = len(pts)
        ids = [sk.addGeometry(Part.LineSegment(Vector(pts[i][0], pts[i][1], 0), Vector(pts[(i + 1) % n][0], pts[(i + 1) % n][1], 0)), False) for i in range(n)]
        for i in range(n):
            sk.addConstraint(Sketcher.Constraint("Coincident", ids[i], 2, ids[(i + 1) % n], 1))
        for i in range(n - 1):
            d = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            sk.addConstraint(Sketcher.Constraint("Horizontal" if abs(d[1]) < 1e-9 else "Vertical", ids[i]))
        sk.addConstraint(Sketcher.Constraint("PointOnObject", ids[0], 1, -2))
        sk.addConstraint(Sketcher.Constraint("PointOnObject", ids[n - 1], 1, -2))
        sk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, ids[0], 1, 0.0))
        names = {1: ("DistanceX", "R_outer"), 2: ("DistanceY", "h_rim"), 3: ("DistanceX", "R_rim"), 4: ("DistanceY", "t_floor")} if v["rings"] else {1: ("DistanceX", "R_outer"), 2: ("DistanceY", "t_floor")}
        for i, (ctype, name) in names.items():
            bind(sk, sk.addConstraint(Sketcher.Constraint(ctype, -1, 1, ids[i], 1, pts[i][0] if ctype == "DistanceX" else pts[i][1])), name)
    rev = doc.addObject("PartDesign::Revolution", "revolve_profile")
    body.addObject(rev)
    rev.Profile = sk
    rev.ReferenceAxis = (sk, ["V_Axis"])
    rev.Angle = 360
    doc.recompute()
    if v["holes"]:
        hs = doc.addObject("Sketcher::SketchObject", "sk_holes")
        body.addObject(hs)
        hs.AttachmentSupport = [(doc.getObject("XY_Plane") or [o for o in body.Origin.OriginFeatures if "XY" in o.Name][0], "")]
        hs.MapMode = "FlatFace"
        for j, hole in enumerate(v["holes"]):
            add_loop(hs, hole, "hole%d" % j, mpp)
        pk = doc.addObject("PartDesign::Pocket", "pocket_holes")
        body.addObject(pk)
        pk.Profile = hs
        pk.Type = "ThroughAll"
        symmetric(pk)
        doc.recompute()
    return body


def symmetric(pad):
    if hasattr(pad, "SideType"):
        pad.SideType = "Symmetric"
    else:
        pad.Midplane = True


def build(spec, out):
    doc = App.newDocument(spec.get("name", "part"))
    params = [("mm_per_px", spec["mm_per_px"], spec["scale_note"])]
    if spec.get("revolve"):
        body = build_revolve(spec, doc, params)
        views = {}
    elif spec.get("outline"):
        body = build_outline(spec, doc, params)
        views = {}
    else:
        views = spec["views"]
    for vname, v in views.items():
        for i, st in enumerate(v["stations"]):
            params.append(("%s_w%d" % (vname, i), round(st["width"], 3),
                           "%s width of station %d (px units)" % (vname, i)))
        for i, z in enumerate(v["z"]):
            params.append(("%s_z%d" % (vname, i), round(z, 3), "%s station boundary %d (px units)" % (vname, i)))
    if views:
        sheet_with(doc, params)
        body = doc.addObject("PartDesign::Body", "Body")
        body.Label = spec.get("name", "part")
    tools, first = [], None
    rots = {"front": Rotation(Vector(1, 0, 0), 90),
            "side": Rotation(Vector(0, 0, 1), 90).multiply(Rotation(Vector(1, 0, 0), 90))}
    span = max([max(st["width"] for st in v["stations"]) for v in views.values()] or [1]) * spec["mm_per_px"] * 3
    for vname, v in views.items():
        halves = [(st["width"] / 2 * spec["mm_per_px"], "params.%s_w%d / 2 * params.mm_per_px" % (vname, i))
                  for i, st in enumerate(v["stations"])]
        zs = [(z * spec["mm_per_px"], "params.%s_z%d * params.mm_per_px" % (vname, i)) for i, z in enumerate(v["z"])]
        owner = body if first is None else doc.addObject("PartDesign::Body", "%s_tool" % vname)
        sk = elevation(doc, owner, "sk_%s" % vname, halves, zs, rots[vname])
        pad = doc.addObject("PartDesign::Pad", "pad_%s" % vname)
        owner.addObject(pad)
        pad.Profile = sk
        pad.Length = span
        symmetric(pad)
        doc.recompute()
        if first is None:
            first = owner
        else:
            tools.append(owner)
    if tools:
        boo = doc.addObject("PartDesign::Boolean", "housing")
        body.addObject(boo)
        boo.Type = "Common"
        boo.Group = tools
        doc.recompute()
    s = body.Shape
    sketches = {o.Name: {"solve": o.solve(), "dof": o.getDoF() if hasattr(o, "getDoF") else None, "redundant": list(o.RedundantConstraints),
                         "conflicting": list(o.ConflictingConstraints), "geometry": o.GeometryCount, "constraints": o.ConstraintCount}
                for o in doc.Objects if o.TypeId.startswith("Sketcher")}
    doc.saveAs(out)
    if s.isNull():
        report = {"valid": False, "solids": 0, "volume": 0, "bbox": [0, 0, 0], "sketches": sketches}
        print("REPORT " + json.dumps(report))
        return report
    bb = s.BoundBox
    report = {"valid": bool(s.isValid()), "solids": len(s.Solids), "volume": round(s.Volume, 3),
              "bbox": [round(bb.XLength, 3), round(bb.YLength, 3), round(bb.ZLength, 3)], "sketches": sketches}
    if spec.get("stl"):
        import Mesh
        import MeshPart
        m = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.05, AngularDeflection=0.3, Relative=False)
        Mesh.Mesh(m.Topology).write(spec["stl"])
    print("REPORT " + json.dumps(report))
    return report


def build_one(spec_path, out):
    import FreeCAD
    try:
        report = build(json.load(open(spec_path)), out)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report = {"valid": False, "error": str(exc)[:200], "solids": 0, "bbox": [0, 0, 0], "sketches": {}}
    for doc in list(FreeCAD.listDocuments()):
        FreeCAD.closeDocument(doc)
    return report


def batch(list_path):
    for line in open(list_path):
        line = line.strip()
        if not line:
            continue
        spec_path, out = line.split("\t")
        r = build_one(spec_path, out)
        print("BATCH %s %s" % (os.path.splitext(os.path.basename(out))[0], json.dumps(r)), flush=True)


if os.environ.get("P2F_LIST"):
    batch(os.environ["P2F_LIST"])
else:
    try:
        build(json.load(open(os.environ["P2F_SPEC"])), os.environ["P2F_OUT"])
    except Exception:
        import traceback
        traceback.print_exc()
        raise
