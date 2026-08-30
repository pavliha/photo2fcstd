import numpy as np
import trimesh

from photo2fcstd.score import normalise, voxels

GRID = 64


def rasterise(points2d, grid=GRID):
    p = np.asarray(points2d, float)
    p = p - p.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(p.T))
    p = p @ vecs[:, ::-1]
    span = np.ptp(p, axis=0)
    q = np.round(p * ((grid - 3) / max(span.max(), 1e-9))).astype(int)
    q -= q.min(axis=0)
    q = q[(q[:, 0] < grid) & (q[:, 1] < grid)]
    img = np.zeros((grid, grid), bool)
    img[q[:, 0], q[:, 1]] = True
    return img


def flips(img):
    return [img, img[::-1], img[:, ::-1], img[::-1, ::-1], img.T, img.T[::-1], img.T[:, ::-1], img.T[::-1, ::-1]]


def iou2d(a, b):
    return float(np.logical_and(a, b).sum()) / max(int(np.logical_or(a, b).sum()), 1)


def directions(n=64):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def model_silhouettes(mesh, n=64, grid=GRID):
    pts = np.array(sorted(voxels(normalise(mesh))), float)
    out = []
    for d in directions(n):
        d = d / np.linalg.norm(d)
        a = np.array([1.0, 0.0, 0.0]) if abs(d @ np.array([1.0, 0.0, 0.0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, a)
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)
        out.append(rasterise(pts @ np.c_[u, v], grid))
    return out


def photo_silhouette(mask, grid=GRID):
    ys, xs = np.nonzero(mask)
    return rasterise(np.c_[ys, xs], grid)


def consistency(mesh, masks, n=64):
    silhouettes = model_silhouettes(mesh, n)
    total = sum(max(iou2d(photo_silhouette(mask), f) for s in silhouettes for f in flips(s)) for mask in masks)
    return total / max(len(masks), 1)


def best_candidate(candidates, masks, n=64):
    scored = [(consistency(trimesh.load(path), masks, n), mode) for mode, path in candidates.items()]
    value, mode = max(scored)
    return mode, {m: round(v, 4) for v, m in scored}


def demo():
    box = trimesh.creation.box((2, 1, 0.4))
    flat = np.ones((40, 80), bool)
    assert consistency(box, [flat], n=24) > consistency(trimesh.creation.icosphere(radius=1.0), [flat], n=24)
    print("verify self-check ok")


if __name__ == "__main__":
    demo()
