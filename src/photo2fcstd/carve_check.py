"""Validate carving against a known mesh: exact camera poses, rendered silhouettes.

The carve path needs the ChArUco board in the photos, so it cannot be exercised on the
PrintCAD photo sets. This drives it with synthetic views of a truth mesh instead, which
checks the geometry and the code without any capture."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, cv2, trimesh
from photo2fcstd import carve as C
from photo2fcstd.bench import truth_of

W = H = 900
ELEV = (18.0, 34.0, 55.0, 75.0)
F = 1400.0


def K_of():
    return np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def poses(n, radius, elevations=(18.0, 34.0, 55.0, 75.0)):
    elevations = ELEV
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = np.radians(elevations[i % len(elevations)])
        eye = np.array([radius * np.cos(el) * np.cos(az), radius * np.cos(el) * np.sin(az), radius * np.sin(el)])
        f = -eye / np.linalg.norm(eye)
        up = np.array([0, 0, 1.0])
        r = np.cross(f, up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        R = np.stack([r, d, f])
        assert abs(np.linalg.det(R) - 1) < 1e-6, np.linalg.det(R)
        rvec, _ = cv2.Rodrigues(R)
        tvec = (-R @ eye).reshape(3, 1)
        out.append({"rvec": rvec, "tvec": tvec, "K": K_of(), "dist": np.zeros(5)})
    return out


def silhouette(mesh, view):
    uv = C.project(np.asarray(mesh.vertices, float), view)
    img = np.zeros((H, W), np.uint8)
    tri = np.round(uv[np.asarray(mesh.faces)]).astype(np.int32)
    for t in tri:
        cv2.fillConvexPoly(img, t, 1)
    return img > 0


def voxelise(mesh, pitch):
    v = mesh.voxelized(pitch).fill()
    return set(map(tuple, np.round(v.points / pitch).astype(int)))


def one(args):
    part, nviews, voxel = args
    try:
        m = trimesh.load(truth_of(part))
        m.apply_translation(-np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
        radius = float(np.max(m.extents)) * 3.2
        vs = poses(nviews, radius)
        masks = [silhouette(m, v) for v in vs]
        if min(int(mk.sum()) for mk in masks) < 200:
            return part, {"error": "empty silhouette"}
        bnds = [(float(m.bounds[0][i]) - 2, float(m.bounds[1][i]) + 2) for i in range(2)] + [(0.0, float(m.bounds[1][2]) + 2)]
        carved = C.carve(vs, masks, voxel_mm=voxel, bounds=bnds)
        if carved is None:
            return part, {"error": "carve returned None"}
        a = set(map(tuple, np.round(carved["points_mm"] / voxel).astype(int)))
        b = voxelise(m, voxel)
        iou = len(a & b) / max(len(a | b), 1)
        e_true, e_carve = np.sort(m.extents)[::-1], np.sort(carved["extents_mm"])[::-1]
        return part, {"iou": iou, "n_carved": len(a), "n_true": len(b),
                      "ext_err": [float((c - t) / t) for c, t in zip(e_carve, e_true)],
                      "ext_mm": [float(c - t) for c, t in zip(e_carve, e_true)],
                      "thin_mm": float(e_true[-1]), "voxels_thick": float(e_true[-1] / voxel)}
    except Exception as exc:
        return part, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:70])}


def main():
    global ELEV
    nviews = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    if len(sys.argv) > 3: ELEV = tuple(float(x) for x in sys.argv[3].split(","))
    voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    ideal = json.load(open("data/printcad_ideal_sketches_all.json"))
    from photo2fcstd.sketch_score import trustworthy
    parts = [p for p in open("data/printcad_all_ids.txt").read().split() if p in ideal and trustworthy(ideal[p])][:40]
    with Pool(6) as pool:
        res = dict(pool.map(one, [(p, nviews, voxel) for p in parts]))
    ok = [p for p in parts if "error" in res[p] is False or "iou" in res[p]]
    errs = [res[p]["error"] for p in parts if "error" in res[p]]
    v = np.array([res[p]["iou"] for p in ok])
    ext = np.array([res[p]["ext_err"] for p in ok])
    print("carve: %d views, elevations %s" % (nviews, ELEV))
    print("  %d views, %.2f mm voxels, n=%d (%d errors)" % (nviews, voxel, len(ok), len(errs)))
    if errs:
        print("  errors: %s" % list(dict.fromkeys(errs))[:3])
    if len(v):
        print("  volumetric IoU vs truth: mean %.3f  median %.3f  p10 %.3f  min %.3f" % (v.mean(), np.median(v), np.percentile(v, 10), v.min()))
        print("  extent error: %+.1f%% %+.1f%% %+.1f%% (longest to shortest axis)" % tuple(100 * ext.mean(axis=0)))
        mm = np.array([res[p]["ext_mm"] for p in ok])
        print("  extent error in mm: %+.2f %+.2f %+.2f" % tuple(mm.mean(axis=0)))
        thick = [p for p in ok if res[p]["voxels_thick"] >= 4]
        if thick:
            v2 = np.array([res[p]["iou"] for p in thick])
            e2 = np.array([res[p]["ext_err"] for p in thick])
            print("  parts at least 4 voxels thick (n=%d): IoU %.3f  extent %+.1f%% %+.1f%% %+.1f%%"
                  % (len(thick), v2.mean(), *(100 * e2.mean(axis=0))))
        thin = [p for p in ok if res[p]["voxels_thick"] < 4]
        print("  parts thinner than 4 voxels: %d (median true thickness %.2f mm)"
              % (len(thin), np.median([res[p]["thin_mm"] for p in thin]) if thin else 0))


def demo():
    box = trimesh.creation.box((20.0, 12.0, 4.0))
    box.apply_translation(-np.array([0.0, 0.0, box.bounds[0][2]]))
    vs = poses(14, float(np.max(box.extents)) * 3.2, (6.0, 18.0, 35.0, 60.0))
    masks = [silhouette(box, v) for v in vs]
    carved = C.carve(vs, masks, voxel_mm=0.25,
                     bounds=[(float(box.bounds[0][i]) - 1, float(box.bounds[1][i]) + 1) for i in range(2)]
                            + [(0.0, float(box.bounds[1][2]) + 1)])
    assert carved is not None, "carve returned nothing for a box"
    got = np.sort(carved["extents_mm"])[::-1]
    want = np.sort(box.extents)[::-1]
    err = np.abs(got - want)
    assert err.max() < 1.5, (got, want)
    print("carve self-check ok: box %s mm carved as %s mm" % (np.round(want, 2), np.round(got, 2)))


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
