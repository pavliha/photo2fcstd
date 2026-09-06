import sys, json, os
import FreeCAD, numpy as np
doc = FreeCAD.openDocument(os.environ["FC_IN"])
fuse = [o for o in doc.Objects if o.TypeId == "Part::MultiFuse"]
bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
solids = [o for o in doc.Objects if getattr(o, "Shape", None) and o.Shape.Solids]
sh = (fuse[-1] if fuse else bodies[-1] if bodies else max(solids, key=lambda o: o.Shape.Volume)).Shape
v, t = sh.tessellate(0.3)
np.savez(os.environ["FC_MESH"], V=np.array([[p.x, p.y, p.z] for p in v]), T=np.array(t))
sheets = [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]
rows = []
for s in sheets:
    for r in range(1, 80):
        try:
            a = s.get("A%d" % r)
        except Exception:
            break
        vals = []
        for c in "BCD":
            try:
                vals.append(s.get("%s%d" % (c, r)))
            except Exception:
                vals.append("")
        rows.append([a] + vals)
json.dump({"bbox": [sh.BoundBox.XLength, sh.BoundBox.YLength, sh.BoundBox.ZLength], "volume": sh.Volume, "sheet": rows}, open(os.environ["FC_INFO"], "w"))
