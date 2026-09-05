import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))


def fcstd_to_stl(fcstd, stl):
    scr = tempfile.NamedTemporaryFile("w", suffix=".py", dir=ROOT, delete=False)
    scr.write(
        "import FreeCAD\n"
        "doc=FreeCAD.openDocument(%r)\n"
        "fuse=[o for o in doc.Objects if o.TypeId=='Part::MultiFuse']\n"
        "bodies=[o for o in doc.Objects if o.TypeId=='PartDesign::Body']\n"
        "s=[o for o in doc.Objects if getattr(o,'Shape',None) and o.Shape.Solids]\n"
        "sh=(fuse[-1] if fuse else bodies[-1] if bodies else max(s,key=lambda o:o.Shape.Volume)).Shape\n"
        "sh.exportStl(%r)\n" % (fcstd, stl))
    scr.close()
    try:
        subprocess.run([FREECAD, scr.name], capture_output=True, text=True, timeout=300)
    finally:
        os.remove(scr.name)
    return os.path.exists(stl)


def solid_iou(truth_path, fcstd):
    import trimesh
    from photo2fcstd import score
    stl = fcstd + ".stl"
    if not fcstd_to_stl(fcstd, stl):
        return None
    iou, _ = score.best_iou(trimesh.load(truth_path), trimesh.load(stl))
    return float(iou)


def main(parts):
    from photo2fcstd import bench, cli, design, recognise
    rows = []
    for part in parts:
        photos = bench.photos_of(part)[:3]
        truth = bench.truth_of(part)
        d = tempfile.mkdtemp(prefix="rvf_")
        flat_spec, _ = recognise.route(photos, name=part, rec={"single_extrusion": True, "face_photo_index": 0})
        sp = os.path.join(d, "flat.spec.json"); json.dump(flat_spec, open(sp, "w"))
        r = cli.freecad_build(sp, os.path.join(d, "flat.FCStd"))
        flat_iou = solid_iou(truth, os.path.join(d, "flat.FCStd")) if r["valid"] else None
        rec = recognise._recognise_program_live(photos)
        res = design.design(photos, os.path.join(d, "routed.FCStd"), rec=rec)
        if res.get("model", 1) is None:
            routed_tier, routed_iou = "refused", None
        else:
            routed_tier = res["tier"]
            routed_iou = solid_iou(truth, os.path.join(d, "routed.FCStd"))
        rows.append({"part": part, "flat_mode": flat_spec.get("mode"), "flat_iou": flat_iou,
                     "rec_revolve": rec.get("revolve"), "routed_tier": routed_tier, "routed_iou": routed_iou})
        print("ROW %s flat[%s] IoU=%s | routed[%s] IoU=%s" % (part, flat_spec.get("mode"), flat_iou, routed_tier, routed_iou), flush=True)
    json.dump(rows, open(os.path.join(ROOT, "runs", "route_vs_flat.json"), "w"), indent=1)
    f = [r["flat_iou"] for r in rows if r["flat_iou"] is not None]
    g = [r["routed_iou"] for r in rows if r["routed_iou"] is not None]
    print("SUMMARY n=%d flat mean solid IoU %.3f | routed mean %.3f (built %d, refused %d)"
          % (len(rows), np.mean(f) if f else -1, np.mean(g) if g else -1, len(g), sum(r["routed_tier"] == "refused" for r in rows)))


if __name__ == "__main__":
    rows = [r for r in json.load(open(os.path.join(ROOT, "runs", "gate_bench.json"))) if "f1" in r and r.get("valid")]
    rows.sort(key=lambda r: r["f1"])
    main([r["part"] for r in rows[:int(sys.argv[1]) if len(sys.argv) > 1 else 8]])
