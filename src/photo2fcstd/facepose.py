import os

import cv2
import numpy as np

_MODEL = {}
TILT_RECTIFY_DEG = float(os.environ.get("P2F_TILT_RECTIFY_DEG", "30"))
RAW_VERIFY_MAX = float(os.environ.get("P2F_RAW_VERIFY_MAX", "0.90"))


def _model():
    import torch
    if "m" not in _MODEL:
        from vggt.models.vggt import VGGT
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL["m"] = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
        _MODEL["dev"] = dev
    return _MODEL["m"], _MODEL["dev"]


def _plane(P, tol, iters=600, rng=np.random.default_rng(0)):
    best = (0, None, 0.0)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n); d = -n @ a
        k = int((np.abs(P @ n + d) < tol).sum())
        if k > best[0]:
            best = (k, n, d)
    inl = np.abs(P @ best[1] + best[2]) < tol
    c = P[inl].mean(0); w, v = np.linalg.eigh(np.cov((P[inl] - c).T)); n = v[:, 0] / np.linalg.norm(v[:, 0])
    return n, -n @ c, inl


def rectify(photos, masks, res=800):
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model, dev = _model()
    images = load_and_preprocess_images(photos).to(dev)
    with torch.no_grad():
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    E = extr.squeeze(0).float().cpu().numpy(); K = intr.squeeze(0).float().cpu().numpy()
    Wp = pred["world_points"].squeeze(0).float().cpu().numpy(); WC = pred["world_points_conf"].squeeze(0).float().cpu().numpy()
    H, W = images.shape[-2:]
    M = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    keep = WC > np.quantile(WC, 0.3)
    allp = Wp[keep].reshape(-1, 3)
    if len(allp) > 300000:
        allp = allp[np.random.default_rng(0).choice(len(allp), 300000, replace=False)]
    dn, dd, _ = _plane(allp, 0.01 * np.ptp(allp, 0).max())
    cams = np.array([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(E))])
    if np.median(cams @ dn + dd) < 0:
        dn, dd = -dn, -dd
    obj = Wp[keep & M].reshape(-1, 3)
    obj = obj[(obj @ dn + dd) > 0.005 * np.ptp(allp, 0).max()]
    if len(obj) < 500:
        return None
    if len(obj) > 200000:
        obj = obj[np.random.default_rng(1).choice(len(obj), 200000, replace=False)]
    fn, fd, inl = _plane(obj, 0.004 * np.ptp(obj, 0).max())
    if fn @ dn < 0:
        fn, fd = -fn, -fd
    face = obj[inl]; o = face.mean(0)
    u = np.cross(fn, [1.0, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(fn, [0, 1.0, 0])
    u /= np.linalg.norm(u); v = np.cross(fn, u)
    uv = np.column_stack([(obj - o) @ u, (obj - o) @ v])
    mn = np.quantile(uv, 0.002, 0) - 0.02 * np.ptp(uv, 0).max(); s = (res - 8) / (np.ptp(uv, 0).max() * 1.06)
    grid = np.array([[1 / s, 0, mn[0] - 4 / s], [0, 1 / s, mn[1] - 4 / s], [0, 0, 1]])
    rect, tilts = [], []
    for i in range(len(photos)):
        R, t = E[i, :3, :3], E[i, :3, 3]
        Hp = K[i] @ np.column_stack([R @ u, R @ v, R @ o + t]) @ grid
        rect.append(cv2.warpPerspective(M[i].astype(np.uint8), np.linalg.inv(Hp), (res, res), flags=cv2.INTER_NEAREST) > 0)
        tilts.append(float(np.degrees(np.arccos(min(abs((R.T @ np.array([0, 0, 1.0])) @ fn), 1.0)))))
    agree = [float(np.mean([_iou(rect[i], rect[j]) for j in range(len(rect)) if j != i])) for i in range(len(rect))]
    return {"rectified": rect, "tilts_deg": tilts, "agreement": agree, "face_inliers": float(inl.mean())}


def _iou(a, b):
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def spec_from_mask(mask, name, source="rectified"):
    from scipy import ndimage
    from photo2fcstd import analysis, spec as spec_mod
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    lab, k = ndimage.label(m)
    if k > 1:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, k + 1))) + 1)
    m = ndimage.binary_fill_holes(m)
    loops = spec_mod.traced_outline(analysis.view_from_mask(m))
    if not loops:
        return None
    return {"name": name, "mode": "plan", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: set from one caliper reading", "views": {},
            "outline": {"source": source, "loops": loops, "depth_px": 0.2 * max(np.ptp(np.nonzero(m)[1]), 1),
                        "depth_note": "depth inferred (untrusted)", "depth_trusted": False},
            "revolve": None, "stl": None, "measured": []}


def maybe_rectify(photos, spec, name="part"):
    from photo2fcstd import trace
    from photo2fcstd.trace import segment_photo
    if not spec.get("outline") or len(photos) < 2:
        return spec, {"applied": False, "reason": "no outline or fewer than 2 photos"}
    if spec.get("mode") != "plan":
        return spec, {"applied": False, "reason": "mode %s keeps its own construction" % spec.get("mode")}
    was = trace.RECOVER_DARK; trace.RECOVER_DARK = False
    try:
        masks = [segment_photo(p) for p in photos]
        info = rectify(photos, masks)
    except Exception as e:
        return spec, {"applied": False, "reason": "facepose failed: %s" % str(e)[:80]}
    finally:
        trace.RECOVER_DARK = was
    if info is None:
        return spec, {"applied": False, "reason": "no object above the desk"}
    src = spec["outline"].get("source")
    chosen = photos.index(src) if src in photos else int(np.argmin(info["tilts_deg"]))
    tilt = info["tilts_deg"][chosen]
    if tilt <= TILT_RECTIFY_DEG:
        return spec, {"applied": False, "reason": "chosen view tilt %.0f deg <= %.0f" % (tilt, TILT_RECTIFY_DEG), "tilts": info["tilts_deg"]}
    best = int(np.argmax(info["agreement"]))
    if info["agreement"][best] <= AGREEMENT_MIN:
        return spec, {"applied": False, "reason": "rectified views disagree (agreement %.2f <= %.2f)" % (info["agreement"][best], AGREEMENT_MIN),
                      "tilts": info["tilts_deg"], "agreement": info["agreement"]}
    new = spec_from_mask(info["rectified"][best], name, source="rectified:%s" % os.path.basename(photos[best]))
    if new is None:
        return spec, {"applied": False, "reason": "rectified mask traced to nothing", "tilts": info["tilts_deg"]}
    new["outline"]["depth_px"] = spec["outline"].get("depth_px", new["outline"]["depth_px"])
    new["outline"]["depth_note"] = spec["outline"].get("depth_note", "")
    new["outline"]["tilt_note"] = "chosen view tilted %.0f deg; outline traced from view %d rectified by the VGGT face pose" % (tilt, best)
    return new, {"applied": True, "tilt": tilt, "view": best, "tilts": info["tilts_deg"], "agreement": info["agreement"],
                 "face_inliers": info["face_inliers"], "mask": info["rectified"][best]}
