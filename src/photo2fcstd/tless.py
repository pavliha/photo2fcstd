"""Drive the carve path from T-LESS, which ships calibrated poses and CAD models.

No board, no shooting: every image already has cam_K and cam_R/cam_t, so this is the
capture problem solved for us. `carve_check.py` proves the geometry on synthetic views;
this proves it on real photographs.
"""
import json
import os

import numpy as np

from photo2fcstd import settings

ROOT = os.path.expanduser(os.environ.get("P2F_TLESS", os.path.join(settings.bop_dir(), "tless")))


def scene_dir(obj_id, split="train_primesense"):
    return os.path.join(ROOT, split, "%06d" % obj_id)


def model_path(obj_id):
    return os.path.join(ROOT, "models_cad", "obj_%06d.ply" % obj_id)


def views_of(obj_id, every=1, limit=None, split="train_primesense"):
    """(view dict for carve.project, mask path) for each image, in the object's frame."""
    import cv2
    d = scene_dir(obj_id, split)
    cams = json.load(open(os.path.join(d, "scene_camera.json")))
    gts = json.load(open(os.path.join(d, "scene_gt.json")))
    keys = sorted(cams, key=int)[::every]
    keys = keys[:limit] if limit else keys
    out = []
    for k in keys:
        gt = gts[k][0]
        R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"], float).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        view = {"rvec": rvec, "tvec": t, "K": np.array(cams[k]["cam_K"], float).reshape(3, 3),
                "dist": np.zeros(5), "elev": cams[k].get("elev")}
        out.append((view, os.path.join(d, "mask_visib", "%06d_000000.png" % int(k))))
    return out


def depths_of(pairs, obj_id, split="train_primesense"):
    """Depth maps in millimetres, aligned with `pairs`."""
    import cv2
    d = scene_dir(obj_id, split)
    cams = json.load(open(os.path.join(d, "scene_camera.json")))
    out = []
    for view, mask_path in pairs:
        key = os.path.basename(mask_path).split("_")[0]
        raw = cv2.imread(os.path.join(d, "depth", key + ".png"), cv2.IMREAD_UNCHANGED)
        out.append(None if raw is None else raw.astype(float) * cams[str(int(key))]["depth_scale"])
    return out


def fuse_object(obj_id, every=40, limit=None, voxel_mm=0.5, stride=2, erode_px=None):
    """Point cloud from the posed depth maps, in the object frame."""
    import trimesh
    from photo2fcstd import fuse as F
    pairs = views_of(obj_id, every=every, limit=limit)
    masks = masks_of(pairs)
    depths = depths_of(pairs, obj_id)
    keep = [i for i, m in enumerate(masks) if m is not None and m.sum() > 200 and depths[i] is not None]
    if len(keep) < 3:
        return None, None
    pts = F.fuse([pairs[i][0] for i in keep], [depths[i] for i in keep],
                 [masks[i] for i in keep], stride=stride, voxel_mm=voxel_mm,
                 erode_px=F.ERODE_PX if erode_px is None else erode_px)
    return pts, trimesh.load(model_path(obj_id))


def carve_object_with_depth(obj_id, every=60, voxel_mm=0.8, margin=6.0, free_space_mm=6.0, min_votes=2):
    import trimesh
    from photo2fcstd import fuse as F
    pairs = views_of(obj_id, every=every)
    masks = masks_of(pairs)
    depths = depths_of(pairs, obj_id)
    keep = [i for i, m in enumerate(masks) if m is not None and m.sum() > 200]
    if len(keep) < 3:
        return None, None
    mesh = trimesh.load(model_path(obj_id))
    lo, hi = mesh.bounds
    bounds = [(float(lo[i]) - margin, float(hi[i]) + margin) for i in range(3)]
    carved = F.carve_with_depth([pairs[i][0] for i in keep], [masks[i] for i in keep],
                                [depths[i] for i in keep], voxel_mm=voxel_mm,
                                bounds=bounds, free_space_mm=free_space_mm, min_votes=min_votes)
    return carved, mesh


def masks_of(pairs):
    import cv2
    return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127 for _, p in pairs]


def carve_object(obj_id, every=40, limit=None, voxel_mm=0.5, margin=6.0):
    """Carve one object from its posed photographs. Returns the carved dict."""
    import trimesh
    from photo2fcstd import carve as C
    pairs = views_of(obj_id, every=every, limit=limit)
    views = [v for v, _ in pairs]
    masks = masks_of(pairs)
    keep = [i for i, m in enumerate(masks) if m is not None and m.sum() > 200]
    views = [views[i] for i in keep]
    masks = [masks[i] for i in keep]
    if len(views) < 3:
        return None, None
    mesh = trimesh.load(model_path(obj_id))
    lo, hi = mesh.bounds
    bounds = [(float(lo[i]) - margin, float(hi[i]) + margin) for i in range(3)]
    return C.carve(views, masks, voxel_mm=voxel_mm, bounds=bounds), mesh


def demo():
    carved, mesh = carve_object(1, every=120, voxel_mm=1.0)
    assert carved is not None, "nothing carved"
    got = np.sort(carved["extents_mm"])[::-1]
    want = np.sort(mesh.extents)[::-1]
    print("tless self-check: obj 1 from %d views -> %s mm, CAD says %s mm"
          % (carved["views"], np.round(got, 1), np.round(want, 1)))
    assert np.abs(got - want).max() < 8.0, (got, want)


if __name__ == "__main__":
    demo()
