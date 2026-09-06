import math
import os

import cv2
import numpy as np
import torch

RES = 128
DT = torch.float32
DEV = torch.device(os.environ.get("P2F_MV_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
FILL = 0.7
TAU = 1.0
MAX_VERTS = 40
FIT_MIN = float(os.environ.get("P2F_MV_FIT_MIN", "0.9"))
TILT_MIN_DEG = float(os.environ.get("P2F_MV_TILT_MIN", "8"))
INIT_TILTS = (0.0, 25.0, -25.0, 45.0, -45.0)
RESIDUAL = float(os.environ.get("P2F_MV_RESIDUAL", "2.0"))
MANHATTAN = float(os.environ.get("P2F_MV_MANHATTAN", "0.1"))


def focal_35(path):
    try:
        from PIL import Image
        return float(Image.open(path).getexif().get_ifd(0x8769).get(41989) or 0) or 27.0
    except Exception:
        return 27.0


def _rodrigues(r):
    th = r.norm().clamp_min(1e-9)
    k = r / th
    K = torch.stack([torch.stack([torch.zeros((), dtype=r.dtype, device=r.device), -k[2], k[1]]),
                     torch.stack([k[2], torch.zeros((), dtype=r.dtype, device=r.device), -k[0]]),
                     torch.stack([-k[1], k[0], torch.zeros((), dtype=r.dtype, device=r.device)])])
    return torch.eye(3, dtype=r.dtype, device=r.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _sdf(poly, G):
    a = poly[:, :, None, :]
    e = torch.roll(poly, -1, 1)[:, :, None, :] - a
    w = G[None, None] - a
    t = ((w * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-9)).clamp(0, 1)
    d = (w - t[..., None] * e).norm(dim=-1).amin(1)
    ay, by = a[..., 1], a[..., 1] + e[..., 1]
    gx, gy = G[None, None, :, 0], G[None, None, :, 1]
    cond = (ay <= gy) != (by <= gy)
    xint = a[..., 0] + (gy - ay) / torch.where(e[..., 1].abs() < 1e-12, torch.full_like(e[..., 1], 1e-12), e[..., 1]) * e[..., 0]
    inside = ((cond & (gx < xint)).sum(1) % 2) == 1
    return torch.where(inside, -d, d)


def _project(P, depth, rvec, t, f, c):
    R = torch.stack([_rodrigues(r) for r in rvec])
    top = torch.cat([P, torch.zeros(*P.shape[:2], 1, dtype=P.dtype, device=P.device)], 2)
    bot = torch.cat([P, -depth[:, None, None].expand(P.shape[0], P.shape[1], 1)], 2)
    X = torch.cat([top, bot], 1) @ R.transpose(1, 2) + t[:, None]
    uv = f[:, None, None] * X[..., :2] / X[..., 2:3].clamp_min(1e-3) + c[:, None]
    return uv[:, :P.shape[1]], uv[:, P.shape[1]:]


def silhouette(P, depth, rvec, t, f, c, G):
    B, N = P.shape[:2]
    top, bot = _project(P, depth, rvec, t, f, c)
    quads = torch.stack([top, torch.roll(top, -1, 1), torch.roll(bot, -1, 1), bot], 2).reshape(B * N, 4, 2)
    faces = _sdf(torch.cat([top, bot]), G).reshape(2, B, -1).amin(0)
    s = torch.minimum(faces, _sdf(quads, G).reshape(B, N, -1).amin(1))
    return torch.sigmoid(-s / TAU)


def _soft_iou(a, b):
    return (a * b).sum(-1) / (a + b - a * b).sum(-1).clamp_min(1e-6)


def _prep(mask, f35):
    from scipy import ndimage
    mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = FILL * RES / max(y1 - y0, x1 - x0)
    small = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))),
                       interpolation=cv2.INTER_AREA) > 0
    out = np.zeros((RES, RES), bool)
    py, px = (RES - small.shape[0]) // 2, (RES - small.shape[1]) // 2
    out[py:py + small.shape[0], px:px + small.shape[1]] = small
    H, W = mask.shape
    c = np.array([(W / 2 - x0) * s + px, (H / 2 - y0) * s + py])
    f = f35 * max(H, W) / 36.0 * s
    return out, f, c, (x0, y0, s, px, py)


def _init_poly(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cs, key=cv2.contourArea)
    eps = 0.5
    poly = cv2.approxPolyDP(cnt, eps, True)
    while len(poly) > MAX_VERTS:
        eps *= 1.4
        poly = cv2.approxPolyDP(cnt, eps, True)
    return poly.reshape(-1, 2).astype(np.float64)


def _norm(P):
    Q = P - P.mean(0)
    return Q / (Q.max(0).values - Q.min(0).values).max().clamp_min(1e-6)


def _manhattan(P):
    e = torch.roll(P, -1, 0) - P
    ang = torch.atan2(e[:, 1], e[:, 0])
    return (e.norm(dim=1) * torch.sin(2 * ang).pow(2)).sum()


def _grid():
    y, x = torch.meshgrid(torch.arange(RES, dtype=DT, device=DEV), torch.arange(RES, dtype=DT, device=DEV), indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2) + 0.5


def _pose_init(m, f, c, tilt_deg, axis):
    ys, xs = np.nonzero(m)
    span = max(np.ptp(xs), np.ptp(ys)) + 1
    tz = f / span
    cu, cv = xs.mean() + 0.5, ys.mean() + 0.5
    r = np.zeros(3); r[axis] = math.radians(tilt_deg)
    return torch.tensor(r, dtype=DT, device=DEV), torch.tensor([(cu - c[0]) / f * tz, (cv - c[1]) / f * tz, tz], dtype=DT, device=DEV)


def _T(x):
    return torch.tensor(np.asarray(x), dtype=DT, device=DEV)


def _unproject(poly_px, r, t, f, c):
    R = _rodrigues(r).cpu().numpy(); tt = t.cpu().numpy()
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], tt])
    q = np.linalg.solve(H, np.column_stack([poly_px, np.ones(len(poly_px))]).T).T
    P = q[:, :2] / q[:, 2:3]
    m = P.mean(0); k = float(np.ptp(P, 0).max())
    P = (P - m) / k
    t2 = (R @ np.array([m[0], m[1], 0.0]) + tt) / k
    return torch.tensor(P, dtype=DT, device=DEV), torch.tensor(t2, dtype=DT, device=DEV)


def _pose_search(P0, preps, targets, fs, cs, G, views, lr, iters=60):
    cands = [(i, tilt, axis) for i in views for tilt in INIT_TILTS for axis in ((0, 1) if tilt else (0,))]
    inits = [_pose_init(preps[i][0], preps[i][1], preps[i][2], tilt, axis) for i, tilt, axis in cands]
    r = torch.stack([q for q, _ in inits]).requires_grad_(True); t = torch.stack([q for _, q in inits]).requires_grad_(True)
    vi = torch.tensor([i for i, _, _ in cands], device=DEV)
    d0 = torch.full((len(cands),), 0.2, dtype=DT, device=DEV)
    opt = torch.optim.Adam([r, t], lr=lr)
    per = torch.zeros(len(cands), device=DEV)
    for _ in range(iters):
        opt.zero_grad()
        for i in views:
            j = vi == i
            soft = silhouette(P0[None].expand(int(j.sum()), -1, -1), d0[j], r[j], t[j], fs[vi[j]], cs[vi[j]], G)
            l = 1 - _soft_iou(soft, targets[vi[j]])
            l.sum().backward(); per[j] = l.detach()
        opt.step()
    best = [int(min((j for j in range(len(cands)) if cands[j][0] == i), key=lambda j: float(per[j]))) for i in views]
    return r.detach()[best], t.detach()[best], float(per[best].sum())


def fit(masks, f35s, ref=0, iters=200, lr=0.02):
    torch.manual_seed(0)
    preps = [_prep(m, f) for m, f in zip(masks, f35s)]
    G = _grid()
    targets = torch.stack([_T(p[0]).reshape(-1) for p in preps])
    fs, cs = _T([p[1] for p in preps]), _T([p[2] for p in preps])
    poly_px = _init_poly(preps[ref][0])
    others = [i for i in range(len(preps)) if i != ref]
    best = None
    for tilt in INIT_TILTS:
        for axis in ((0, 1) if tilt else (0,)):
            r0, t0 = _pose_init(preps[ref][0], preps[ref][1], preps[ref][2], tilt, axis)
            P0, t0 = _unproject(poly_px, r0, t0, preps[ref][1], preps[ref][2])
            ro, to, cost = _pose_search(P0, preps, targets, fs, cs, G, others, lr)
            if best is None or cost < best[0]:
                best = (cost, P0, r0, t0, ro, to)
    _, P0, r0, t0, ro, to = best
    order = sorted(range(len(preps)), key=lambda i: (i != ref, i))
    r = torch.stack([r0] + list(ro)).clone().requires_grad_(True); t = torch.stack([t0] + list(to)).clone().requires_grad_(True)
    inv = torch.tensor([order.index(i) for i in range(len(preps))], device=DEV)
    A = torch.eye(2, dtype=DT, device=DEV).requires_grad_(True)
    resid = torch.zeros_like(P0).requires_grad_(True)
    depth = torch.tensor(math.log(0.2), dtype=DT, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{"params": [resid], "lr": lr / 4}, {"params": [A, depth, r, t], "lr": lr}])
    for _ in range(iters):
        opt.zero_grad()
        Pn = _norm(P0 @ A.T + resid)
        soft = silhouette(Pn[None].expand(len(preps), -1, -1), depth.exp().expand(len(preps)), r[inv], t[inv], fs, cs, G)
        loss = (1 - _soft_iou(soft, targets)).sum()
        (loss + MANHATTAN * _manhattan(Pn) + RESIDUAL * resid.pow(2).sum()).backward(); opt.step()
    with torch.no_grad():
        Pn = _norm(P0 @ A.T + resid)
        soft = silhouette(Pn[None].expand(len(preps), -1, -1), depth.exp().expand(len(preps)), r[inv], t[inv], fs, cs, G)
        ious = _soft_iou((soft > 0.5).to(DT), targets).cpu().numpy().tolist()
        Rs = [_rodrigues(q).cpu().numpy() for q in r[inv]]
        tilts = [float(math.degrees(math.acos(min(1.0, abs(R[2, 2]))))) for R in Rs]
        Hs = [_plane_to_image(R, q.cpu().numpy(), p[1], p[2], p[3]) for R, q, p in zip(Rs, t[inv], preps)]
    return {"P": Pn.cpu().numpy(), "depth": float(depth.exp().detach().cpu()), "ious": ious, "tilts": tilts, "H": Hs, "ref": ref}


def _plane_to_image(R, t, f, c, crop):
    x0, y0, s, px, py = crop
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    Hres = K @ np.column_stack([R[:, 0], R[:, 1], t])
    back = np.array([[1 / s, 0, x0 - px / s], [0, 1 / s, y0 - py / s], [0, 0, 1.0]])
    return back @ Hres


def rectified_mask(res_fit, mask_ref, size=800):
    P = res_fit["P"]
    lo, hi = P.min(0), P.max(0)
    s = (size - 16) / (hi - lo).max()
    A = np.array([[s, 0, 8 - lo[0] * s], [0, s, 8 - lo[1] * s], [0, 0, 1.0]])
    H = res_fit["H"][res_fit["ref"]] @ np.linalg.inv(A)
    from scipy import ndimage
    warped = cv2.warpPerspective(mask_ref.astype(np.uint8), np.linalg.inv(H), (size, size), flags=cv2.INTER_NEAREST) > 0
    holes = ndimage.binary_fill_holes(warped) & ~warped
    face = np.zeros((size, size), np.uint8)
    cv2.fillPoly(face, [((P - lo) * s + 8).astype(np.int32)], 1)
    return (face > 0) & ~holes, res_fit["depth"] * s


def maybe_rectify(photos, spec, name="part"):
    from photo2fcstd import trace
    from photo2fcstd.facepose import spec_from_mask
    from photo2fcstd.trace import segment_photo
    if not spec.get("outline") or len(photos) < 2:
        return spec, {"applied": False, "reason": "no outline or fewer than 2 photos"}
    if spec.get("mode") != "plan":
        return spec, {"applied": False, "reason": "mode %s keeps its own construction" % spec.get("mode")}
    if spec["outline"].get("warning", "").startswith("the photos look at this part edge-on"):
        return spec, {"applied": False, "reason": "edge-on part: the sketch is its cross-section"}
    was = trace.RECOVER_DARK; trace.RECOVER_DARK = False
    try:
        masks = [segment_photo(p) for p in photos]
    finally:
        trace.RECOVER_DARK = was
    src = spec["outline"].get("source")
    ref = photos.index(src) if src in photos else 0
    try:
        res = fit(masks, [focal_35(p) for p in photos], ref=ref)
    except Exception as e:
        return spec, {"applied": False, "reason": "multiview fit failed: %s" % str(e)[:80]}
    info = {"fit_iou": res["ious"], "tilts": res["tilts"], "depth": res["depth"], "view": ref}
    mask, depth_px = rectified_mask(res, masks[ref])
    new = spec_from_mask(mask, name, source="multiview:%s" % os.path.basename(photos[ref]))
    if new is None:
        return spec, {**info, "applied": False, "reason": "fitted outline traced to nothing"}
    new["outline"]["depth_px"] = float(depth_px)
    new["outline"]["depth_note"] = "depth from the three-view silhouette fit (%.2f of the face extent)" % res["depth"]
    new["outline"]["tilt_note"] = "outline fitted jointly to %d views; reference view tilted %.0f deg" % (len(photos), res["tilts"][ref])
    if min(res["ious"]) < FIT_MIN:
        return spec, {**info, "applied": False, "reason": "fit IoU %.2f < %.2f" % (min(res["ious"]), FIT_MIN), "candidate": new}
    if res["tilts"][ref] < TILT_MIN_DEG:
        return spec, {**info, "applied": False, "reason": "reference view already square (%.0f deg)" % res["tilts"][ref], "candidate": new}
    return new, {**info, "applied": True, "tilt": res["tilts"][ref], "mask": mask}
