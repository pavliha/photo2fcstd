import json
import os

import numpy as np

from photo2fcstd import cli, recognise

TEMPLATES = {}


def register(cls):
    def deco(fn):
        TEMPLATES[cls] = fn
        return fn
    return deco


@register("fan_guard")
def _fan_guard(rec, photos, out, frame_w_mm=80.0, depth_json=None, traced=True):
    path, params = recognise.design_fan(photos, out, rec=rec, frame_w_mm=frame_w_mm, depth_json=depth_json, traced=traced)
    return {"tier": "template", "part_class": "fan_guard", "out": path,
            "params": params, "sketches_clean": True, "valid": True}


THRESHOLD = {"template": 0.8, "revolve": 0.8, "assemble": 0.6, "board": 0.8}


@register("bottle")
def _bottle(rec, photos, out, cloud_npz=None, **kw):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools"))
    import cloud_primitives as cp
    if cloud_npz:
        params, ledger, st = cp.fit_revolve(cloud_npz)
        contain = st["contain_fraction"]
    else:
        prof = recognise.revolve_spec(photos, rec, name=_stem(out))["revolve"]["profile"][1:-1]
        params, ledger = cp.fit_revolve_profile([(r, h) for r, h in prof], "side photo")
        contain = None
    cp.build_revolve_template(params, ledger, out)
    return {"tier": "template", "part_class": "bottle", "out": out, "params": {**params, "_ledger": ledger},
            "sketches_clean": True, "valid": True, "contain_fraction": contain}


CONTAIN_THRESHOLD = 0.9


def design(photos, out, rec=None, **kw):
    rec = rec if rec is not None else recognise._recognise_program_live(photos)
    cls = rec.get("part_class")
    board = _board_carve(photos)
    if board is not None:
        return _design_from_board(board, photos, out, rec, cls, **kw)
    if cls in TEMPLATES:
        res = TEMPLATES[cls](rec, photos, out, **kw)
    elif rec.get("revolve"):
        res = _build(recognise.revolve_spec(photos, rec, name=_stem(out)), out, "revolve", cls)
    elif rec.get("single_extrusion"):
        spec, _ = recognise.route(photos, name=_stem(out), rec=rec)
        res = {**_build(spec, out, "assemble", cls), "tilt": {"applied": False, "reason": "disabled"}}
        if os.environ.get("P2F_FACEPOSE", "1") == "1" and res.get("valid") and spec.get("mode") == "plan":
            from photo2fcstd import facepose
            raw = verify(out, photos, rec)
            if raw["silhouette_iou"] is not None and raw["silhouette_iou"] < facepose.RAW_VERIFY_MAX:
                spec2, tilt_info = _rectifier().maybe_rectify(photos, spec, name=_stem(out))
                if tilt_info.get("applied"):
                    res = {**_build(spec2, out, "assemble", cls), "tilt": {**tilt_info, "raw_verify_iou": raw["silhouette_iou"]}}
                else:
                    res["tilt"] = tilt_info
            else:
                res["tilt"] = {"applied": False, "reason": "raw build already matches the photo (%.2f)" % raw["silhouette_iou"]}
    else:
        return _refuse(out, "no template for this part and it is not a revolve or a flat extrusion",
                       ["add a template for this class, or shoot a slow low 16-view orbit for the carve path"])
    if res.get("valid") is False:
        return _refuse(out, "build produced no valid solid", ["reshoot square to the face"])
    if res.get("contain_fraction") is not None:          # built from a 3D cloud: gate on containment, not a 2D view
        if res["contain_fraction"] < CONTAIN_THRESHOLD:
            return _refuse(out, "solid contains only %.0f%% of the measured cloud" % (100 * res["contain_fraction"]),
                           ["shoot a slower, lower orbit"])
        return {**res, "verify": {"contain_fraction": res["contain_fraction"], "confidence": "ok"}}
    tilt = res.get("tilt", {})
    v = verify(out, photos, rec, threshold=THRESHOLD[res["tier"]], mask=tilt.get("mask") if tilt.get("applied") else None)
    if "tilt" in res:
        res["tilt"] = {k: val for k, val in tilt.items() if k != "mask"}
    if v["confidence"] != "ok":
        return _refuse(out, "silhouette does not match the photo (IoU %s < %s)"
                       % (v["silhouette_iou"], THRESHOLD[res["tier"]]), v["advice"])
    return {**res, "verify": v}


def _board_carve(photos):
    from photo2fcstd import carve
    from photo2fcstd.errors import CaptureError
    if len(photos) < carve.MIN_POSED_VIEWS:
        return None
    try:
        return carve.from_photos(photos)
    except CaptureError:
        return None


def _design_from_board(carved, photos, out, rec, cls, **kw):
    import tempfile
    from photo2fcstd import carve
    ext = np.sort(carved["extents_mm"])[::-1]
    if cls == "fan_guard":
        dj = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"short_over_long": float(ext[1] / ext[0]), "height_over_long": float(carved["top_mm"] / ext[0]), "source": "board hull"}, dj); dj.close()
        res = TEMPLATES[cls](rec, photos, out, frame_w_mm=float(ext[0]), depth_json=dj.name, traced=kw.get("traced", True))
        res["params"]["_ledger"]["frame_w"] = "measured (board hull)"
    elif rec.get("revolve"):
        spec = carve.revolve_from_carve(carved, _stem(out), rec)
        if not spec["revolve"]["holes"] and "bore" in (rec.get("openings") or []):
            ratio, how = recognise.bore_ratio(photos)
            if ratio:
                R = spec["revolve"]["profile"][1][0]
                spec["revolve"]["holes"].append({"type": "circle", "cx": 0.0, "cy": 0.0, "r": round(R * ratio, 2), "source": how})
        res = _build(spec, out, "board", cls)
    else:
        face = rec.get("face_photo_index")
        axis = carve.axis_facing(carved["board_views"][carved["sources"].index(photos[face])]) \
            if face is not None and photos[face] in carved["sources"] else None
        spec = carve.spec_from_carve(carved, _stem(out), axis=axis, views=carved["board_views"], masks=carved["board_masks"])
        res = _build(spec, out, "board", cls)
    if res.get("valid") is False:
        return _refuse(out, "build produced no valid solid", ["reshoot with the target fully in frame"])
    mask = carve.occupancy(carved, axis=2)
    v = verify(out, photos, rec, threshold=THRESHOLD["board"], mask=mask)
    if v["confidence"] != "ok":
        return _refuse(out, "model does not match the carved hull (IoU %s < %s)" % (v["silhouette_iou"], THRESHOLD["board"]), v["advice"])
    return {**res, "board": {"views": carved["views"], "extents_mm": [round(float(x), 2) for x in carved["extents_mm"]], "top_mm": round(float(carved["top_mm"]), 2),
                             "min_elevation_deg": round(float(carved["min_elevation_deg"]), 1)}, "verify": v}


def _rectifier():
    from photo2fcstd import facepose
    return facepose


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


def verify(out, photos, rec, threshold=None, mask=None):
    import numpy as np
    from photo2fcstd.trace import segment_photo, upright_mask
    from photo2fcstd import trace
    if mask is None:
        face = photos[int(rec.get("face_photo_index", 0))]
        was = trace.RECOVER_DARK
        trace.RECOVER_DARK = False          # gross outline only; verify must not inherit build state
        try:
            mask, _ = upright_mask(segment_photo(face))
        finally:
            trace.RECOVER_DARK = was
    else:
        mask, _ = upright_mask(mask)
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
        "fuse=[o for o in doc.Objects if o.TypeId=='Part::MultiFuse']\n"
        "bodies=[o for o in doc.Objects if o.TypeId=='PartDesign::Body']\n"
        "s=[o for o in doc.Objects if getattr(o,'Shape',None) and o.Shape.Solids]\n"
        "sh=(fuse[-1] if fuse else bodies[-1] if bodies else max(s,key=lambda o:o.Shape.Volume)).Shape\n"
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
    from scipy import ndimage
    k = max(3, int(0.03 * max(mask.shape)))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    mask = ndimage.binary_fill_holes(mask)        # the contract is the outer outline; holes are the ledger's job
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


def _aligned_iou_masks(a, b):
    ca = _canon(a); b0 = _canon(b)
    best = 0.0
    for bb in (b0, b0[::-1], b0[:, ::-1], np.rot90(b0), np.rot90(b0)[::-1]):
        bb = _canon(bb)
        best = max(best, float(np.logical_and(ca, bb).sum() / (np.logical_or(ca, bb).sum() or 1)))
    return best


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
    ap.add_argument("--depth-json", default=None, help="vggt_depth.py result; gives the enclosure its measured depth")
    ap.add_argument("--cloud", default=None, help="vggt_depth.py *_poses.npz; cloud-fitted templates (bottle)")
    ap.add_argument("--masks", default=None, help="directory of <photo stem>.png masks to use instead of segmenting (video path: hands removed)")
    a = ap.parse_args(argv)
    if a.masks:
        os.environ["P2F_MASK_DIR"] = a.masks
    rec = recognise._recognise_program_live(a.photos)
    print("recognised:", json.dumps(rec))
    cls = rec.get("part_class")
    res = design(a.photos, a.out, rec=rec, frame_w_mm=a.frame_mm, depth_json=a.depth_json) if cls == "fan_guard" \
        else design(a.photos, a.out, rec=rec, cloud_npz=a.cloud) if cls == "bottle" else design(a.photos, a.out, rec=rec)
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
