import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def fit_plane_ransac(P, iters=600, tol=None, rng=np.random.default_rng(0)):
    tol = tol if tol is not None else 0.01 * np.ptp(P, 0).max()
    best_n, best_d, best_in = None, 0.0, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n); d = -n @ a
        inl = int((np.abs(P @ n + d) < tol).sum())
        if inl > best_in:
            best_n, best_d, best_in = n, d, inl
    inl = np.abs(P @ best_n + best_d) < tol
    c = P[inl].mean(0); w, v = np.linalg.eigh(np.cov((P[inl] - c).T)); n = v[:, 0] / np.linalg.norm(v[:, 0])
    return n, -n @ c, inl


def run_part(model, part_dir, out_dir, res=640):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    names = sorted(glob.glob(os.path.join(part_dir, "*.jpg")))
    masks = [os.path.join(part_dir, "masks", os.path.splitext(os.path.basename(p))[0] + ".png") for p in names]
    images = load_and_preprocess_images(names).to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    E = extr.squeeze(0).float().cpu().numpy(); K = intr.squeeze(0).float().cpu().numpy()
    Hh, Ww = images.shape[-2:]
    Wp = pred["world_points"].squeeze(0).float().cpu().numpy(); WC = pred["world_points_conf"].squeeze(0).float().cpu().numpy()
    M = (load_and_preprocess_images(masks)[:, 0] > 0.5).cpu().numpy()
    keep = WC > np.quantile(WC, 0.3)
    allp = Wp[keep].reshape(-1, 3)
    if len(allp) > 300000:
        allp = allp[np.random.default_rng(0).choice(len(allp), 300000, replace=False)]
    dn, dd, _ = fit_plane_ransac(allp)                       # desk
    cams = np.array([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(E))])
    if np.median(cams @ dn + dd) < 0:
        dn, dd = -dn, -dd
    obj = Wp[keep & M].reshape(-1, 3)
    obj = obj[(obj @ dn + dd) > 0.005 * np.ptp(allp, 0).max()]
    if len(obj) < 500:
        return None
    if len(obj) > 200000:
        obj = obj[np.random.default_rng(1).choice(len(obj), 200000, replace=False)]
    fn, fd, inl = fit_plane_ransac(obj, tol=0.004 * np.ptp(obj, 0).max())   # the visible top face
    if fn @ dn < 0:
        fn, fd = -fn, -fd
    face = obj[inl]
    u = np.cross(fn, [1.0, 0, 0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(fn, [0, 1.0, 0]) / np.linalg.norm(np.cross(fn, [0, 1.0, 0]))
    v = np.cross(fn, u)
    uv = np.column_stack([(obj - face.mean(0)) @ u, (obj - face.mean(0)) @ v])          # whole object projected square-on
    uvf = np.column_stack([(face - face.mean(0)) @ u, (face - face.mean(0)) @ v])
    mn = np.quantile(uv, 0.002, 0) - 0.02 * np.ptp(uv, 0).max(); span = np.ptp(uv, 0).max() * 1.06
    s = (res - 8) / span
    img_all = np.zeros((res, res), np.uint8); img_face = np.zeros((res, res), np.uint8)
    for arr, img in ((uv, img_all), (uvf, img_face)):
        q = ((arr - mn) * s + 4).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < res) & (q[:, 1] >= 0) & (q[:, 1] < res)
        img[q[ok, 1], q[ok, 0]] = 255
    part = os.path.basename(part_dir.rstrip("/"))
    import cv2
    o = face.mean(0)
    grid = np.array([[1 / s, 0, mn[0] - 4 / s], [0, 1 / s, mn[1] - 4 / s], [0, 0, 1]])   # rectified pixel -> plane (a, b)
    for i in range(len(names)):
        R, t = E[i, :3, :3], E[i, :3, 3]
        Hp = K[i] @ np.column_stack([R @ u, R @ v, R @ o + t])                       # plane (a,b,1) -> image pixel
        Hfull = Hp @ grid                                                             # rectified pixel -> image pixel
        mimg = (M[i] * 255).astype(np.uint8)
        rect = cv2.warpPerspective(mimg, np.linalg.inv(Hfull), (res, res), flags=cv2.INTER_NEAREST)
        Image.fromarray(rect).save(os.path.join(out_dir, "%s_rect_v%d.png" % (part, i)))
    Image.fromarray(img_all).save(os.path.join(out_dir, part + "_topview.png"))
    Image.fromarray(img_face).save(os.path.join(out_dir, part + "_faceview.png"))
    tilts = [float(np.degrees(np.arccos(abs((E[i, :3, :3].T @ np.array([0, 0, 1.0])) @ fn)))) for i in range(len(E))]
    return {"part": part, "views": len(names), "face_inliers": float(inl.mean()), "obj_points": int(len(obj)),
            "face_normal_vs_desk_deg": float(np.degrees(np.arccos(min(abs(fn @ dn), 1.0)))), "view_tilts_deg": tilts}


def main(root, out_dir):
    from vggt.models.vggt import VGGT
    os.makedirs(out_dir, exist_ok=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        try:
            r = run_part(model, d, out_dir)
        except Exception as e:
            r = {"part": os.path.basename(d), "error": str(e)[:120]}
        rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(out_dir, "face_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
