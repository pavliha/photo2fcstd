import json
import math
import os

import FreeCAD as App
import Part
import Sketcher
from FreeCAD import Vector

P = os.environ.get("P2F_OUT", os.path.expanduser("~/Downloads/revolve.FCStd"))
_m = json.load(open(os.environ["P2F_PARAMS"])) if os.environ.get("P2F_PARAMS") else {}
_ledger = _m.get("_ledger", {})
doc = App.newDocument("revolve")

_defaults = dict(body_r=17.5, body_h=70.0, shoulder_h=8.0, neck_r=9.0, neck_h=15.0, cap_r=11.0, cap_h=12.0)
_notes = dict(body_r="body radius", body_h="body height (straight wall)", shoulder_h="shoulder taper height",
              neck_r="neck radius", neck_h="neck height", cap_r="cap radius", cap_h="cap height")
sh = doc.addObject("Spreadsheet::Sheet", "params")
sh.set("A1", "param"); sh.set("B1", "value"); sh.set("C1", "meaning"); sh.set("D1", "source")
rows = [(k, float(_m.get(k, _defaults[k])), _notes[k]) for k in _defaults]
for i, (a, v, n) in enumerate(rows, start=2):
    sh.set("A%d" % i, a); sh.set("B%d" % i, str(v)); sh.set("C%d" % i, n); sh.setAlias("B%d" % i, a)
    sh.set("D%d" % i, _ledger.get(a, "default"))
doc.recompute()
G = {a: v for a, v, _ in rows}

body = doc.addObject("PartDesign::Body", "Body")
sk = body.newObject("Sketcher::SketchObject", "sk_profile")
sk.AttachmentSupport = [(doc.getObject("XZ_Plane"), "")]; sk.MapMode = "FlatFace"

h0 = G["body_h"]; h1 = h0 + G["shoulder_h"]; h2 = h1 + G["neck_h"]; h3 = h2 + G["cap_h"]
capped = abs(G["cap_r"] - G["neck_r"]) > 1e-6 and G["cap_h"] > 1e-6
if capped:
    pts = [Vector(0, 0, 0), Vector(G["body_r"], 0, 0), Vector(G["body_r"], h0, 0), Vector(G["neck_r"], h1, 0),
           Vector(G["neck_r"], h2, 0), Vector(G["cap_r"], h2, 0), Vector(G["cap_r"], h3, 0), Vector(0, h3, 0)]
    horiz, vert = (0, 4, 6), (1, 3, 5, 7)
else:
    pts = [Vector(0, 0, 0), Vector(G["body_r"], 0, 0), Vector(G["body_r"], h0, 0), Vector(G["neck_r"], h1, 0),
           Vector(G["neck_r"], h3, 0), Vector(0, h3, 0)]
    horiz, vert = (0, 4), (1, 3, 5)
ids = [sk.addGeometry(Part.LineSegment(pts[i], pts[(i + 1) % len(pts)]), False) for i in range(len(pts))]
for i in range(len(ids)):
    sk.addConstraint(Sketcher.Constraint("Coincident", ids[i], 2, ids[(i + 1) % len(ids)], 1))
sk.addConstraint(Sketcher.Constraint("Coincident", ids[0], 1, -1, 1))          # profile starts on the axis origin
for i in horiz:
    sk.addConstraint(Sketcher.Constraint("Horizontal", ids[i]))
for i in vert:
    sk.addConstraint(Sketcher.Constraint("Vertical", ids[i]))

def bind(kind, gid, value, alias, expr):
    c = sk.addConstraint(Sketcher.Constraint(kind, -1, 1, gid, 2, value)); sk.renameConstraint(c, alias); sk.setExpression("Constraints." + alias, expr)

bind("DistanceX", ids[0], G["body_r"], "body_r", "params.body_r")
bind("DistanceY", ids[1], h0, "body_h", "params.body_h")
bind("DistanceX", ids[2], G["neck_r"], "neck_r", "params.neck_r")
bind("DistanceY", ids[2], h1, "h1", "params.body_h + params.shoulder_h")
if capped:
    bind("DistanceY", ids[3], h2, "h2", "params.body_h + params.shoulder_h + params.neck_h")
    bind("DistanceX", ids[4], G["cap_r"], "cap_r", "params.cap_r")
    bind("DistanceY", ids[5], h3, "h3", "params.body_h + params.shoulder_h + params.neck_h + params.cap_h")
else:
    bind("DistanceY", ids[3], h3, "h3", "params.body_h + params.shoulder_h + params.neck_h + params.cap_h")
doc.recompute()
print("profile fully constrained:", sk.FullyConstrained)

rev = body.newObject("PartDesign::Revolution", "revolve")
rev.Profile = sk; rev.ReferenceAxis = (sk, ["V_Axis"]); rev.Angle = 360
doc.recompute()
final = body
for i, b in enumerate(_m.get("bosses") or []):
    cyl = Part.makeCylinder(float(b["r"]), float(b["len"]), Vector(*b["base"]), Vector(*b["axis"]))
    obj = doc.addObject("Part::Feature", "boss_%d" % i); obj.Shape = cyl
    fuse = doc.addObject("Part::MultiFuse", "part"); fuse.Shapes = [final, obj]; doc.recompute(); final = fuse
    sh.set("A%d" % (len(rows) + 2 + i), "boss_%d" % i); sh.set("C%d" % (len(rows) + 2 + i), "measured cylinder r=%.1f len=%.1f" % (b["r"], b["len"]))
    sh.set("D%d" % (len(rows) + 2 + i), _ledger.get("boss", "measured"))
doc.recompute()
bb = final.Shape.BoundBox
print("BBOX", [round(x, 2) for x in (bb.XLength, bb.YLength, bb.ZLength)], "valid", final.Shape.isValid(),
      "vol", round(final.Shape.Volume, 1), "solids", len(final.Shape.Solids))
doc.saveAs(P)
print("SAVED", P)
