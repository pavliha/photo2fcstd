import itertools
import sys

import os

import numpy as np
import trimesh

from photo2fcstd.settings import cache_dir
VERSION = 2
PITCH = 1 / 64
FINE = 1 / 128


def normalise(mesh, pca=False):
    m = mesh.copy()
    if pca:
        m.apply_transform(m.principal_inertia_transform)
    m.apply_translation(-m.bounds.mean(axis=0))
    m.apply_scale(1.0 / np.max(m.extents))
    return m


def cached_voxels(path, pca=False, pitch=PITCH):
    import hashlib
    key = hashlib.sha1(("%s|%d|%s|%s|%d" % (os.path.abspath(path), os.stat(path).st_size, pca, pitch, VERSION)).encode()).hexdigest()
    f = os.path.join(cache_dir("voxels"), key + ".npy")
    if os.path.exists(f):
        return set(map(tuple, np.load(f)))
    v = voxels(normalise(trimesh.load(path), pca), pitch)
    os.makedirs(cache_dir("voxels"), exist_ok=True)
    np.save(f, np.array(sorted(v), dtype=np.int16) if v else np.zeros((0, 3), np.int16))
    return v


def voxels(mesh, pitch=PITCH):
    v = mesh.voxelized(pitch).fill()
    pts = np.round((v.points + 0.5) / pitch).astype(int)
    return set(map(tuple, pts))


def signed_perms():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), int)
            for i, p in enumerate(perm):
                m[i, p] = signs[i]
            if round(np.linalg.det(m)) == 1:
                out.append((perm, signs))
    return out


def centred(pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return pts - ((lo + hi) // 2)


def rotate_points(pts, perm, signs):
    q = pts[:, list(perm)].copy()
    for i, sg in enumerate(signs):
        if sg < 0:
            q[:, i] = -q[:, i]
    return centred(q)


def as_points(vox):
    return centred(np.array(sorted(vox), dtype=np.int32)) if vox else np.zeros((0, 3), np.int32)


def rotations():
    axes = list(itertools.permutations(range(3)))
    signs = list(itertools.product([1, -1], repeat=3))
    mats = [np.array([[s[i] if j == p[i] else 0 for j in range(3)] for i in range(3)])
            for p in axes for s in signs]
    return [m for m in mats if abs(np.linalg.det(m) - 1) < 1e-9]


KEY_BASE = 4096


def keys(pts):
    q = np.asarray(pts, np.int64) + KEY_BASE // 2
    return np.sort(q[:, 0] * KEY_BASE * KEY_BASE + q[:, 1] * KEY_BASE + q[:, 2])


def iou_keys(a, b):
    if not len(a) or not len(b):
        return 0.0
    i = np.searchsorted(a, b)
    i[i >= len(a)] = 0
    inter = int((a[i] == b).sum())
    return inter / max(len(a) + len(b) - inter, 1)


def iou(a, b):
    return len(a & b) / max(len(a | b), 1)


def aligned(cand_mesh, R, pca=False):
    c = normalise(cand_mesh, pca).copy()
    c.apply_transform(np.vstack([np.hstack([R, [[0], [0], [0]]]), [0, 0, 0, 1]]))
    return normalise(c)


def best_alignment(ref_mesh, cand_mesh, fine=PITCH, ref_path=None):
    refs = {pca: as_points(cached_voxels(ref_path, pca) if ref_path else voxels(normalise(ref_mesh, pca))) for pca in (False, True)}
    ref_keys = {pca: keys(refs[pca]) for pca in refs}
    scored = []
    for pca in (False, True):
        base = as_points(voxels(normalise(cand_mesh, pca)))
        for j, rot in enumerate(signed_perms()):
            scored.append((iou_keys(ref_keys[pca], keys(rotate_points(base, *rot))), j, rot, pca))
    raw = scored[0][0]
    best, _, R, pca = max(scored, key=lambda t: t[0])
    if fine == PITCH:
        cand = rotate_points(as_points(voxels(normalise(cand_mesh, pca))), *R)
        return best, raw, (R, pca), set(map(tuple, refs[pca])), set(map(tuple, cand))
    ref_fine = as_points(voxels(normalise(ref_mesh, pca), fine))
    cand_fine = rotate_points(as_points(voxels(normalise(cand_mesh, pca), fine)), *R)
    return iou_keys(keys(ref_fine), keys(cand_fine)), raw, (R, pca), set(map(tuple, ref_fine)), set(map(tuple, cand_fine))


def best_iou(ref, cand_mesh):
    ref_path = ref if isinstance(ref, str) else None
    ref_mesh = trimesh.load(ref) if ref_path else ref
    b, raw, _, _, _ = best_alignment(ref_mesh, cand_mesh, ref_path=ref_path)
    return b, raw


def demo():
    box = trimesh.creation.box((2, 1, 0.5))
    same, _ = best_iou(box, trimesh.creation.box((1, 2, 0.5)))
    assert same > 0.97, same
    cyl, _ = best_iou(box, trimesh.creation.cylinder(0.5, 2.0))
    assert cyl < 0.9, cyl
    print("shape_iou self-check ok: rotated box %.2f, cylinder-vs-box %.2f" % (same, cyl))


def main(argv):
    if len(argv) < 2:
        demo()
        raise SystemExit("\nphoto2fcstd-score <reference.stl> <candidate.stl>   (scale-free, best of 24 axis rotations)")
    b, raw = best_iou(argv[0], trimesh.load(argv[1]))
    print("IoU best-aligned %.3f   as-is %.3f" % (b, raw))


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
