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


def main(frames_dir, out_json):
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
    if len(P) > 400000:
        P = P[np.random.default_rng(0).choice(len(P), 400000, replace=False)]
    n, d, frac = fit_plane_ransac(P)
    h = P @ n + d
    if np.median(h) > 0:
        n, d, h = -n, -d, -h
    h = -h
    scale = np.ptp(P, 0).max()
    fg = P[h > 0.02 * scale]
    if len(fg) < 200:
        raise SystemExit("no foreground above the desk plane")
    hf = fg @ n + d
    hf = -hf if np.median(hf) < 0 else hf
    top = hf[hf >= np.quantile(hf, 0.8)]
    depth_p50_top, depth_p90 = float(np.median(top)), float(np.quantile(hf, 0.9))
    Q = fg - np.outer(fg @ n + d, n)
    c = Q.mean(0)
    u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    ext = np.ptp((Q - c) @ vt.T, 0)
    length = float(ext[:2].max())
    res = {"frames": len(names), "plane_inlier_frac": float(frac), "length_units": length,
           "depth_units_top_median": depth_p50_top, "depth_units_p90": depth_p90,
           "depth_ratio": depth_p50_top / length, "depth_ratio_p90": depth_p90 / length,
           "fg_points": int(len(fg))}
    np.savez(os.path.splitext(out_json)[0] + "_poses.npz",
             extrinsic=extr.squeeze(0).float().cpu().numpy(),
             intrinsic=intr.squeeze(0).float().cpu().numpy(),
             plane_n=n, plane_d=d, names=np.array(names))
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
