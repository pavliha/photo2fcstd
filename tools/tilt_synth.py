"""Training rows for a tilt head that has actually seen PrintCAD parts.

The T-LESS head reaches 4.8 degrees held out by object and 34 on PrintCAD, with its gate refusing
every PrintCAD image - the same cross-dataset collapse the depth head showed, caught this time
before shipping. PrintCAD photographs have no tilt truth, so the labels have to come from renders
of the meshes at a chosen tilt.

Renders are not photographs, so the point of the augmentation is the gap itself: light direction,
albedo, contrast, blur and sensor noise vary per sample, because a head fitted on one fixed
Lambertian look will describe that look rather than the tilt. The background is zeroed, which
production also does, and which the T-LESS probe showed costs nothing (+0.05 deg).
"""
import json, os, sys

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import bench, embed, sketch_score as SS  # noqa: E402
import tilt_close as T  # noqa: E402

TILTS = (0.0, 5.0, 10.0, 16.0, 22.0, 30.0)


def lit(mesh, view, rng):
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    uv = T.C.project(V, view)
    R = view["R"]
    cam = (R @ V.T + view["tvec"]).T
    depth = cam[F].mean(1)[:, 2]
    fn = np.asarray(mesh.face_normals, float) @ R.T
    light = rng.normal(size=3)
    light[2] = -abs(light[2]) - 0.4
    light /= np.linalg.norm(light)
    ambient, gain = rng.uniform(20, 70), rng.uniform(120, 210)
    shade = ambient + gain * np.clip(np.abs(fn @ light), 0, 1)
    tint = rng.uniform(0.75, 1.0, 3)
    img = np.zeros((T.H, T.W, 3), np.float32)
    mask = np.zeros((T.H, T.W), np.uint8)
    tri = np.round(uv[F]).astype(np.int32)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, tri[i], tuple(float(shade[i] * t) for t in tint))
        cv2.fillConvexPoly(mask, tri[i], 1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
    img += rng.normal(0, rng.uniform(1, 7), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask > 0


def crop_of(img, mask, rng):
    ys, xs = np.nonzero(mask)
    pad = int(rng.uniform(0.10, 0.22) * max(np.ptp(xs), np.ptp(ys)))
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(T.W, xs.max() + pad), min(T.H, ys.max() + pad))
    cut = img * mask[:, :, None]
    return Image.fromarray(cut[box[1]:box[3], box[0]:box[2]]).resize((embed.SIZE, embed.SIZE))


def main(limit=500, out="data/tilt_printcad.npz", seed=0):
    rng = np.random.default_rng(seed)
    ideal = T.IDEAL
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p]) and T.face_normal(ideal[p]) is not None][:limit]
    imgs, Y, G = [], [], []
    for k, part in enumerate(parts):
        n = T.face_normal(ideal[part])
        try:
            m = trimesh.load(bench.truth_of(part))
            assert len(m.faces)
        except Exception:
            continue
        m.apply_translation(-m.bounds.mean(0))
        rad = float(np.max(m.extents)) * rng.uniform(3.2, 5.0)
        for t in TILTS:
            v = T.view_at(n, t, float(rng.uniform(0, 360)), rad)
            img, mask = lit(m, v, rng)
            if mask.sum() < 300:
                continue
            imgs.append(crop_of(img, mask, rng))
            nc = v["R"] @ n
            Y.append(nc * np.sign(nc[2] if nc[2] != 0 else 1))
            G.append(part)
        if (k + 1) % 100 == 0:
            print("  %d parts, %d views" % (k + 1, len(imgs)))
    X = np.concatenate([embed.embed_images(imgs[i:i + 16]) for i in range(0, len(imgs), 16)])
    np.savez(os.path.join(ROOT, out), obj=X, normal=np.array(Y),
             part=np.array([hash(g) % 10 ** 8 for g in G]), name=np.array(G))
    print("\n  wrote %s: %d views over %d parts" % (out, len(X), len(set(G))))


if __name__ == "__main__":
    main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]])
