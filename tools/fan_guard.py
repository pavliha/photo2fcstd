import math, os, FreeCAD as App, Part, Sketcher
from FreeCAD import Vector

import json
P = os.environ.get("P2F_OUT", os.path.expanduser("~/Downloads/fan/fan.FCStd"))
_pj = os.environ.get("P2F_PARAMS")
_m = json.load(open(_pj)) if _pj else {}
doc = App.newDocument("fan")

sh = doc.addObject("Spreadsheet::Sheet", "params")
_defaults = dict(frame_w=80.0, corner_r=4.0, plate_t=4.0, bore_d=76.0,
                 mount_pitch=71.5, mount_d=4.5, rings=4, wire_w=2.4)
_notes = dict(frame_w="outer square edge (mm)", corner_r="outer corner radius",
              plate_t="guard plate / grille thickness", bore_d="central opening diameter",
              mount_pitch="screw hole centre-to-centre", mount_d="screw hole diameter",
              rings="concentric grille rings (edit + rerun)", wire_w="grille wire / spoke width")
rows = [(k, float(_m.get(k, _defaults[k])), _notes[k]) for k in _defaults]
for i, (a, v, n) in enumerate(rows, start=2):
    sh.set("A%d" % i, a); sh.set("B%d" % i, str(v)); sh.set("C%d" % i, n); sh.setAlias("B%d" % i, a)
doc.recompute()
G = {a: float(v) for a, v, _ in rows}
half = G["frame_w"] / 2; r = G["corner_r"]; t = G["plate_t"]

body = doc.addObject("PartDesign::Body", "Body")

sk = body.newObject("Sketcher::SketchObject", "sk_frame")
sk.AttachmentSupport = [(doc.getObject("XY_Plane"), "")]; sk.MapMode = "FlatFace"
d = half - r
cs = [Vector(d, d, 0), Vector(-d, d, 0), Vector(-d, -d, 0), Vector(d, -d, 0)]
angs = [(0, 90), (90, 180), (180, 270), (270, 360)]
arcs = [sk.addGeometry(Part.ArcOfCircle(Part.Circle(c, Vector(0, 0, 1), r),
        math.radians(a0), math.radians(a1)), False) for c, (a0, a1) in zip(cs, angs)]
for i in range(4):
    l = sk.addGeometry(Part.LineSegment(sk.Geometry[arcs[i]].EndPoint,
                                        sk.Geometry[arcs[(i + 1) % 4]].StartPoint), False)
    sk.addConstraint(Sketcher.Constraint("Tangent", arcs[i], 2, l, 1))
    sk.addConstraint(Sketcher.Constraint("Tangent", l, 2, arcs[(i + 1) % 4], 1))
# intent: all corners equal, each centre a half-diagonal out, one driving expr per corner
for i in (1, 2, 3):
    sk.addConstraint(Sketcher.Constraint("Equal", arcs[0], arcs[i]))
rc = sk.addConstraint(Sketcher.Constraint("Radius", arcs[0], r))
sk.renameConstraint(rc, "corner_r"); sk.setExpression("Constraints.corner_r", "params.corner_r")
signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
for a, (sx, sy) in zip(arcs, signs):
    cx = sk.addConstraint(Sketcher.Constraint("DistanceX", -1, 1, a, 3, sx * d))
    sk.setExpression("Constraints[%d]" % cx, "%sparams.frame_w / 2 - %sparams.corner_r" % ("" if sx > 0 else "-", "" if sx > 0 else "-"))
    cy = sk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, a, 3, sy * d))
    sk.setExpression("Constraints[%d]" % cy, "%sparams.frame_w / 2 - %sparams.corner_r" % ("" if sy > 0 else "-", "" if sy > 0 else "-"))
doc.recompute()
print("frame fully constrained:", sk.FullyConstrained)
pad = body.newObject("PartDesign::Pad", "plate")
pad.Profile = sk; pad.Length = t; pad.setExpression("Length", "params.plate_t")
doc.recompute()
print("plate", round(body.Shape.Volume, 1), body.Shape.isValid())

bsk = body.newObject("Sketcher::SketchObject", "sk_bore")
bsk.AttachmentSupport = [(doc.getObject("XY_Plane"), "")]; bsk.MapMode = "FlatFace"
bsk.addGeometry(Part.Circle(Vector(0, 0, 0), Vector(0, 0, 1), G["bore_d"] / 2), False)
bsk.addConstraint(Sketcher.Constraint("Coincident", 0, 3, -1, 1))
bc = bsk.addConstraint(Sketcher.Constraint("Diameter", 0, G["bore_d"]))
bsk.renameConstraint(bc, "bore_d"); bsk.setExpression("Constraints.bore_d", "params.bore_d")
doc.recompute()
print("bore fully constrained:", bsk.FullyConstrained)
bore = body.newObject("PartDesign::Pocket", "bore")
bore.Profile = bsk; bore.Type = "ThroughAll"; bore.Midplane = True
doc.recompute()
print("bore", round(body.Shape.Volume, 1), body.Shape.isValid())

msk = body.newObject("Sketcher::SketchObject", "sk_mounts")
msk.AttachmentSupport = [(doc.getObject("XY_Plane"), "")]; msk.MapMode = "FlatFace"
h = G["mount_pitch"] / 2
ids = [msk.addGeometry(Part.Circle(Vector(sx, sy, 0), Vector(0, 0, 1), G["mount_d"] / 2), False)
       for sx in (-h, h) for sy in (-h, h)]
for i in ids[1:]:
    msk.addConstraint(Sketcher.Constraint("Equal", ids[0], i))
dm = msk.addConstraint(Sketcher.Constraint("Diameter", ids[0], G["mount_d"]))
msk.renameConstraint(dm, "mount_d"); msk.setExpression("Constraints.mount_d", "params.mount_d")
msigns = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
for i, (sx, sy) in zip(ids, msigns):
    cx = msk.addConstraint(Sketcher.Constraint("DistanceX", -1, 1, i, 3, sx * h))
    msk.setExpression("Constraints[%d]" % cx, "%sparams.mount_pitch / 2" % ("" if sx > 0 else "-"))
    cy = msk.addConstraint(Sketcher.Constraint("DistanceY", -1, 1, i, 3, sy * h))
    msk.setExpression("Constraints[%d]" % cy, "%sparams.mount_pitch / 2" % ("" if sy > 0 else "-"))
doc.recompute()
print("mounts fully constrained:", msk.FullyConstrained)
mp = body.newObject("PartDesign::Pocket", "mounts")
mp.Profile = msk; mp.Type = "ThroughAll"; mp.Midplane = True
doc.recompute()
print("mounts", round(body.Shape.Volume, 1), body.Shape.isValid())

# grille: rings + X spokes as a fused solid, then union with the plate via a second body boolean
R = G["bore_d"] / 2; w = G["wire_w"]; n = int(G["rings"])
solids = []
radii = [R * (1 - 0.9 * k / n) for k in range(n)]
for rr in radii:
    ring = Part.makeCylinder(rr + w / 2, t).cut(Part.makeCylinder(rr - w / 2, t))
    solids.append(ring)
for a in (math.radians(45), math.radians(135)):
    bar = Part.makeBox(2 * R, w, t, Vector(-R, -w / 2, 0))
    bar.Placement = App.Placement(Vector(0, 0, 0), App.Rotation(Vector(0, 0, 1), math.degrees(a)))
    solids.append(bar)
grille = solids[0]
for s in solids[1:]:
    grille = grille.fuse(s)
grille = grille.removeSplitter()
gobj = doc.addObject("Part::Feature", "grille_mesh")
gobj.Shape = grille
doc.recompute()

fuse = doc.addObject("Part::MultiFuse", "fan")
fuse.Shapes = [body, gobj]
doc.recompute()
print("grille+plate", round(fuse.Shape.Volume, 1), fuse.Shape.isValid(), "faces", len(fuse.Shape.Faces))

bb = fuse.Shape.BoundBox
print("BBOX", [round(x, 1) for x in (bb.XLength, bb.YLength, bb.ZLength)],
      "frame_constrained", sk.FullyConstrained)
doc.recompute()
doc.saveAs(P)
print("SAVED", P)
