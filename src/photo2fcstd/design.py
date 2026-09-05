import json
import os

from photo2fcstd import cli, recognise

TEMPLATES = {}


def register(cls):
    def deco(fn):
        TEMPLATES[cls] = fn
        return fn
    return deco


@register("fan_guard")
def _fan_guard(rec, photos, out, frame_w_mm=80.0):
    path, params = recognise.design_fan(photos, out, rec=rec, frame_w_mm=frame_w_mm)
    return {"tier": "template", "part_class": "fan_guard", "out": path,
            "params": params, "sketches_clean": True}


def design(photos, out, rec=None, **kw):
    rec = rec if rec is not None else recognise._recognise_program_live(photos)
    cls = rec.get("part_class")
    if cls in TEMPLATES:
        return TEMPLATES[cls](rec, photos, out, **kw)
    if rec.get("revolve"):
        spec = recognise.revolve_spec(photos, rec, name=_stem(out))
        return _build(spec, out, "revolve", cls)
    if rec.get("single_extrusion"):
        spec, _ = recognise.route(photos, name=_stem(out), rec=rec)
        return _build(spec, out, "assemble", cls)
    spec = recognise.compile_program(rec, photos, name=_stem(out))
    return _build(spec, out, "general", cls)


def _build(spec, out, tier, cls):
    sp = os.path.splitext(out)[0] + ".spec.json"
    json.dump(spec, open(sp, "w"))
    r = cli.freecad_build(sp, out)
    return {"tier": tier, "part_class": cls, "out": out, "valid": r["valid"],
            "solids": r["solids"], "sketches_clean": False}


def _stem(out):
    return os.path.splitext(os.path.basename(out))[0]


def verify(out, photos, rec):
    import numpy as np
    from photo2fcstd.trace import outline, segment_photo, upright_mask
    face = photos[int(rec.get("face_photo_index", 0))]
    mask, _ = upright_mask(segment_photo(face))
    model = _top_silhouette(out)
    advice = []
    iou = None
    if model is not None:
        a = _fill(outline(mask)[0], mask.shape)
        b = _fill(model, mask.shape)
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum()) or 1.0
        iou = round(inter / union, 3)
        if iou < 0.85:
            advice.append("face silhouette off (IoU %.2f) - reshoot square to the face" % iou)
    if not rec.get("_depth_measured"):
        advice.append("depth is estimated (untrusted) - set params or shoot a 16-view orbit")
    return {"silhouette_iou": iou, "advice": advice,
            "confidence": "low" if (iou is not None and iou < 0.85) else "ok"}


def _top_silhouette(out):
    import subprocess, tempfile
    freecad = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
    scr = tempfile.NamedTemporaryFile("w", suffix=".py", dir=os.path.dirname(os.path.abspath(out)), delete=False)
    npz = out + ".sil.npz"
    scr.write(
        "import FreeCAD, numpy as np\n"
        "doc=FreeCAD.openDocument(%r)\n"
        "s=[o for o in doc.Objects if getattr(o,'Shape',None) and o.Shape.Solids]\n"
        "sh=max(s,key=lambda o:o.Shape.Volume).Shape\n"
        "vs=np.array([[v.X,v.Y,v.Z] for v in sh.Vertexes])\n"
        "ax=np.argsort(np.ptp(vs,0))[-2:]\n"     # two widest axes = the largest visible face
        "np.savez(%r,v=vs[:,ax])\n" % (out, npz))
    scr.close()
    try:
        subprocess.run([freecad, scr.name], capture_output=True, text=True, timeout=120)
        import numpy as np
        d = np.load(npz)
        pts = d["v"]
        import cv2
        hull = cv2.convexHull(pts.astype(np.float32))
        return hull.reshape(-1, 2)
    except Exception:
        return None
    finally:
        for f in (scr.name, npz):
            os.path.exists(f) and os.remove(f)


def _fill(poly, shape):
    import cv2, numpy as np
    img = np.zeros(shape, np.uint8)
    p = np.asarray(poly, float)
    p = p - p.mean(0)
    p = p / (np.abs(p).max() + 1e-9) * 0.45 * min(shape) + np.array(shape[::-1]) / 2
    cv2.fillPoly(img, [p.astype(np.int32)], 1)
    return img > 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="photo2fcstd.design",
                                 description="photos -> a parametric FreeCAD model")
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-mm", type=float, default=80.0, help="longest face edge in mm, for scale")
    a = ap.parse_args(argv)
    rec = recognise._recognise_program_live(a.photos)
    print("recognised:", json.dumps(rec))
    res = design(a.photos, a.out, rec=rec, frame_w_mm=a.frame_mm) if rec.get("part_class") == "fan_guard" \
        else design(a.photos, a.out, rec=rec)
    v = verify(a.out, a.photos, rec)
    print("built:", json.dumps({k: res.get(k) for k in ("tier", "part_class", "out", "valid", "sketches_clean")}))
    print("verify:", json.dumps(v))
    return res


if __name__ == "__main__":
    main()
