import glob
import json
import os
import sys

import numpy as np
import torch


def fit_plane_ransac(P, iters=300, tol=None, rng=np.random.default_rng(0)):
    tol = tol if tol is not None else 0.01 * np.ptp(P, 0).max()
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


def main(frames_dir, out_json, masks_dir=None):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "cuda"
    names = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    images = load_and_preprocess_images(names).to(dev)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    W = pred["world_points"].squeeze(0).float().cpu().numpy()
    C = pred["world_points_conf"].squeeze(0).float().cpu().numpy()
    keep = C > np.quantile(C, 0.5)
    P = W[keep].reshape(-1, 3)
    M = None
    if masks_dir:
        mp = [os.path.join(masks_dir, os.path.splitext(os.path.basename(p))[0] + ".png") for p in names]
        Mt = (load_and_preprocess_images(mp)[:, 0] > 0.5).float()       # same resize/crop as the frames
        Mt = 1 - torch.nn.functional.max_pool2d(1 - Mt[:, None], 7, stride=1, padding=3)[:, 0]  # erode: drop edge flying pixels
        M = (Mt > 0.5).cpu().numpy()
    if len(P) > 400000:
        P = P[np.random.default_rng(0).choice(len(P), 400000, replace=False)]
    n, d, frac = fit_plane_ransac(P)
    E = extr.squeeze(0).float().cpu().numpy()
    cams = np.array([-E[i, :3, :3].T @ E[i, :3, 3] for i in range(len(E))])
    if np.median(cams @ n + d) < 0:          # up = the side the cameras are on
        n, d = -n, -d
    h = P @ n + d
    scale = np.ptp(P, 0).max()
    fg = W[keep & M].reshape(-1, 3) if M is not None else P[h > 0.02 * scale]
    fg = fg[(fg @ n + d) > 0.01 * scale]
    if len(fg) < 200:
        raise SystemExit("no foreground above the desk plane")
    hf = fg @ n + d
    top = hf[hf >= np.quantile(hf, 0.8)]
    depth_p50_top, depth_p90 = float(np.median(top)), float(np.quantile(hf, 0.9))
    Q = fg - np.outer(fg @ n + d, n)
    c = Q.mean(0)
    u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    R = (Q - c) @ vt.T
    ext = np.sort((np.quantile(R, 0.98, 0) - np.quantile(R, 0.02, 0))[:2])[::-1]
    length, short = float(ext[0]), float(ext[1])
    res = {"frames": len(names), "plane_inlier_frac": float(frac),
           "footprint_long": length, "footprint_short": short, "height": depth_p50_top,
           "height_p90": depth_p90, "height_over_long": depth_p50_top / length,
           "short_over_long": short / length,
           "fg_points": int(len(fg)), "masked": M is not None}
    fc = fg.mean(0)
    _, _, axes = np.linalg.svd(fg[np.random.default_rng(0).choice(len(fg), min(len(fg), 200000), replace=False)] - fc,
                               full_matrices=False)
    face_n = axes[2]                                   # smallest-extent axis = the largest face's normal
    view_dirs = np.array([E[i, :3, :3].T @ np.array([0, 0, 1.0]) for i in range(len(E))])
    squareness = np.abs(view_dirs @ face_n)             # 1 = looking straight at the largest face
    res["square_frame"] = os.path.basename(names[int(np.argmax(squareness))])
    res["square_score"] = float(squareness.max())
    np.savez(os.path.splitext(out_json)[0] + "_poses.npz",
             extrinsic=extr.squeeze(0).float().cpu().numpy(),
             intrinsic=intr.squeeze(0).float().cpu().numpy(),
             plane_n=n, plane_d=d, names=np.array(names),
             object_centre=fc, object_axes=axes, squareness=squareness)
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
