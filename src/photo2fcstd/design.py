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
            "params": params, "sketches_clean": True, "valid": True}


THRESHOLD = {"template": 0.8, "revolve": 0.8, "assemble": 0.6}


def design(photos, out, rec=None, **kw):
    rec = rec if rec is not None else recognise._recognise_program_live(photos)
    cls = rec.get("part_class")
    if cls in TEMPLATES:
        res = TEMPLATES[cls](rec, photos, out, **kw)
    elif rec.get("revolve"):
        res = _build(recognise.revolve_spec(photos, rec, name=_stem(out)), out, "revolve", cls)
    elif rec.get("single_extrusion"):
        spec, _ = recognise.route(photos, name=_stem(out), rec=rec)
        res = _build(spec, out, "assemble", cls)
    else:
        return _refuse(out, "no template for this part and it is not a revolve or a flat extrusion",
                       ["add a template for this class, or shoot a slow low 16-view orbit for the carve path"])
    if res.get("valid") is False:
        return _refuse(out, "build produced no valid solid", ["reshoot square to the face"])
    v = verify(out, photos, rec, threshold=THRESHOLD[res["tier"]])
    if v["confidence"] != "ok":
        return _refuse(out, "silhouette does not match the photo (IoU %s < %s)"
                       % (v["silhouette_iou"], THRESHOLD[res["tier"]]), v["advice"])
    return {**res, "verify": v}


def _refuse(out, reason, reshoot):
    for f in (out, os.path.splitext(out)[0] + ".spec.json"):
        os.path.exists(f) and os.remove(f)
    return {"model": None, "reason": reason, "reshoot": reshoot}


def _build(spec, out, tier, cls):
    sp = os.path.splitext(out)[0] + ".spec.json"
    json.dump(spec, open(sp, "w"))
    r = cli.freecad_build(sp, out)
    return {"tier": tier, "part_class": cls, "out": out, "valid": r["valid"],
            "solids": r["solids"], "sketches_clean": False}


def _stem(out):
    return os.path.splitext(os.path.basename(out))[0]


def verify(out, photos, rec, threshold=None):
    import numpy as np
    from photo2fcstd.trace import segment_photo, upright_mask
    from photo2fcstd import trace
    face = photos[int(rec.get("face_photo_index", 0))]
    was = trace.RECOVER_DARK
    trace.RECOVER_DARK = False          # gross outline only; verify must not inherit build state
    try:
        mask, _ = upright_mask(segment_photo(face))
    finally:
        trace.RECOVER_DARK = was
    model = _model_silhouette(out)
    advice = []
    iou = None
    if model is not None:
        iou = round(_aligned_iou(mask, model), 3)
        if threshold is not None and iou < threshold:
            advice.append("face silhouette off (IoU %.2f) - reshoot square to the face" % iou)
    if not rec.get("_depth_measured"):
        advice.append("depth is estimated (untrusted) - set params or shoot a 16-view orbit")
    ok = iou is not None and (threshold is None or iou >= threshold)
    return {"silhouette_iou": iou, "advice": advice, "confidence": "ok" if ok else "low"}


def _model_silhouette(out):
    import subprocess, tempfile
    import numpy as np
    freecad = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
    scr = tempfile.NamedTemporaryFile("w", suffix=".py", dir=os.path.dirname(os.path.abspath(out)), delete=False)
    npz = out + ".sil.npz"
    scr.write(
        "import FreeCAD, numpy as np\n"
        "doc=FreeCAD.openDocument(%r)\n"
        "s=[o for o in doc.Objects if getattr(o,'Shape',None) and o.Shape.Solids]\n"
        "sh=max(s,key=lambda o:o.Shape.Volume).Shape\n"
        "v,t=sh.tessellate(0.5)\n"
        "V=np.array([[p.x,p.y,p.z] for p in v]); T=np.array(t)\n"
        "ax=np.argsort(np.ptp(V,0))[-2:]\n"
        "np.savez(%r,tris=V[T][:,:,ax])\n" % (out, npz))
    scr.close()
    try:
        subprocess.run([freecad, scr.name], capture_output=True, text=True, timeout=120)
        return np.load(npz)["tris"]
    except Exception:
        return None
    finally:
        for f in (scr.name, npz):
            os.path.exists(f) and os.remove(f)


def _rasterize(tris, size=512):
    import cv2
    import numpy as np
    pts = tris.reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    s = (size - 4) / max(float((hi - lo).max()), 1e-9)
    img = np.zeros((size, size), np.uint8)
    for t in ((tris - lo) * s + 2).astype(np.int32):
        cv2.fillConvexPoly(img, t, 1)
    return img > 0


def _canon(mask, size=512):
    import cv2
    import numpy as np
    from photo2fcstd.trace import upright_mask
    k = max(3, int(0.03 * max(mask.shape)))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    m, _ = upright_mask(mask)
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = (size - 4) / max(m.shape)
    r = cv2.resize(m.astype(np.uint8), (max(1, int(m.shape[1] * s)), max(1, int(m.shape[0] * s))),
                   interpolation=cv2.INTER_NEAREST) > 0
    out = np.zeros((size, size), bool)
    y0 = (size - r.shape[0]) // 2; x0 = (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _aligned_iou(mask, tris):
    import numpy as np
    a = _canon(mask)
    b0 = _canon(_rasterize(tris))
    best = 0.0
    for b in (b0, b0[::-1], b0[:, ::-1], np.rot90(b0), np.rot90(b0)[::-1]):
        b = _canon(b)
        inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum() or 1
        best = max(best, inter / union)
    return float(best)


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
    if res.get("model", 1) is None:
        print("REFUSED:", res["reason"])
        for r in res["reshoot"]:
            print("  ->", r)
        return res
    print("built:", json.dumps({k: res.get(k) for k in ("tier", "part_class", "out", "sketches_clean")}))
    print("verify:", json.dumps(res["verify"]))
    return res


if __name__ == "__main__":
    main()
