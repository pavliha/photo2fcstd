import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


BLANK_BG = os.environ.get("FUSE_BLANK_BG", "0") == "1"


def crop_to_object(frames_dir, masks_dir, out_dir, margin=0.6):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "_masks", exist_ok=True)
    names = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    kept = []
    for p in names:
        m = np.array(Image.open(os.path.join(masks_dir, os.path.splitext(os.path.basename(p))[0] + ".png")).convert("L")) > 128
        ys, xs = np.nonzero(m)
        if len(xs) < 500:
            continue
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, m.shape[1])), int(min(cy + side / 2, m.shape[0]))
        img = Image.open(p).convert("RGB").crop((x0, y0, x1, y1))
        if BLANK_BG:
            a = np.array(img); a[~m[y0:y1, x0:x1]] = 128; img = Image.fromarray(a)
        mk = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8))
        b = os.path.basename(p)
        img.save(os.path.join(out_dir, b), quality=95)
        mk.save(os.path.join(out_dir + "_masks", os.path.splitext(b)[0] + ".png"))
        kept.append(os.path.join(out_dir, b))
    return kept


def blank_frames(frames_dir, masks_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    kept = []
    for p in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        m = np.array(Image.open(os.path.join(masks_dir, os.path.splitext(os.path.basename(p))[0] + ".png")).convert("L")) > 128
        a = np.array(Image.open(p).convert("RGB")); a[~m] = 128
        Image.fromarray(a).save(os.path.join(out_dir, os.path.basename(p)), quality=95)
        kept.append(os.path.join(out_dir, os.path.basename(p)))
    return kept


def fit_plane_ransac(P, iters=400, rng=np.random.default_rng(0)):
    tol = 0.01 * np.ptp(P, 0).max()
    best_n, best_d, best_in = None, 0.0, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = int((np.abs(P @ n + d) < tol).sum())
        if inl > best_in:
            best_n, best_d, best_in = n, d, inl
    inl = np.abs(P @ best_n + best_d) < tol
    c = P[inl].mean(0)
    w, v = np.linalg.eigh(np.cov((P[inl] - c).T))
    n = v[:, 0] / np.linalg.norm(v[:, 0])
    return n, -n @ c, inl.mean()


def main(frames_dir, masks_dir, out_json, crop=True):
    import open3d as o3d
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    work = os.path.splitext(out_json)[0] + "_crops"
    names = crop_to_object(frames_dir, masks_dir, work) if crop else blank_frames(frames_dir, masks_dir, work) if BLANK_BG \
        else sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    mdir = work + "_masks" if crop else masks_dir
    dev = "cuda"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    images = load_and_preprocess_images(names).to(dev)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images)
    H, W = images.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    E = extr.squeeze(0).float().cpu().numpy()
    K = intr.squeeze(0).float().cpu().numpy()
    D = pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
    DC = pred["depth_conf"].squeeze(0).float().cpu().numpy()
    Wp = pred["world_points"].squeeze(0).float().cpu().numpy()
    WC = pred["world_points_conf"].squeeze(0).float().cpu().numpy()
    mp = [os.path.join(mdir, os.path.splitext(os.path.basename(p))[0] + ".png") for p in names]
    Mt = (load_and_preprocess_images(mp)[:, 0] > 0.5).float()
    erode = int(os.environ.get("FUSE_ERODE", "5"))
    if erode > 1:
        Mt = 1 - torch.nn.functional.max_pool2d(1 - Mt[:, None], erode, stride=1, padding=erode // 2)[:, 0]
    M = (Mt > 0.5).cpu().numpy()

    keep = WC > np.quantile(WC, 0.5)
    P = Wp[keep].reshape(-1, 3)
    if len(P) > 400000:
        P = P[np.random.default_rng(0).choice(len(P), 400000, replace=False)]
    cams = np.array([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(E))])
    raw_obj = Wp[keep & M].reshape(-1, 3)
    if BLANK_BG:
        oc = raw_obj.mean(0); n = np.linalg.svd(raw_obj[::max(1, len(raw_obj) // 100000)] - oc, full_matrices=False)[2][2]
        n = n if np.median(cams @ n - oc @ n) > 0 else -n
        d = -float(np.quantile(raw_obj @ n, 0.02)); frac = 0.0
    else:
        n, d, frac = fit_plane_ransac(P)
        if np.median(cams @ n + d) < 0:
            n, d = -n, -d
    scale = float(np.ptp(raw_obj, 0).max())

    vdiv = float(os.environ.get("FUSE_VOXEL_DIV", "250"))
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=scale / vdiv, sdf_trunc=scale / (vdiv / 5.0),
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    conf_floor = np.quantile(DC, float(os.environ.get("FUSE_CONF_Q", "0.4")))
    for i in range(len(names)):
        depth = D[i].astype(np.float32).copy()
        depth[~M[i]] = 0.0
        depth[DC[i] < conf_floor] = 0.0
        di = o3d.geometry.Image(depth)
        ci = o3d.geometry.Image(np.zeros((H, W, 3), np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(ci, di, depth_scale=1.0, depth_trunc=scale * 20, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[i, 0, 0]), float(K[i, 1, 1]), float(K[i, 0, 2]), float(K[i, 1, 2]))
        ext = np.eye(4); ext[:3, :4] = E[i]
        vol.integrate(rgbd, intrinsic, ext)
    pcd = vol.extract_point_cloud()
    fused = np.asarray(pcd.points)
    fused = fused if BLANK_BG else fused[(fused @ n + d) > 0.01 * scale]
    if len(fused) > 300000:
        fused = fused[np.random.default_rng(0).choice(len(fused), 300000, replace=False)]
    fc = fused.mean(0)
    _, _, axes = np.linalg.svd(fused - fc, full_matrices=False)
    face_n = axes[2]
    t = (fused - fc) @ face_n
    front = t[t >= np.quantile(t, 0.5)]
    width = float(np.ptp((fused - fc) @ axes[0]))
    spread = float((np.quantile(front, 0.9) - np.quantile(front, 0.1)) / width)
    tr = (raw_obj - raw_obj.mean(0)) @ face_n
    fr = tr[tr >= np.quantile(tr, 0.5)]
    spread_raw = float((np.quantile(fr, 0.9) - np.quantile(fr, 0.1)) / width)
    res = {"frames": len(names), "cropped": crop, "fused_points": int(len(fused)), "raw_points": int(len(raw_obj)),
           "front_spread_over_width_fused": spread, "front_spread_over_width_raw": spread_raw,
           "plane_inlier_frac": float(frac), "voxel_over_width": float((scale / vdiv) / width),
           "conf_q": float(os.environ.get("FUSE_CONF_Q", "0.4")), "erode": erode}
    np.savez_compressed(os.path.splitext(out_json)[0] + "_poses.npz", extrinsic=E, intrinsic=K, plane_n=n, plane_d=d,
                        names=np.array(names), object_centre=fc, object_axes=axes, cloud=fused.astype(np.float32),
                        cloud_raw=raw_obj[np.random.default_rng(1).choice(len(raw_obj), min(len(raw_obj), int(os.environ.get("FUSE_RAW_CAP", "300000"))), replace=False)].astype(np.float32))
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], crop=(sys.argv[4] != "nocrop") if len(sys.argv) > 4 else True)
