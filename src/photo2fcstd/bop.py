import glob
import json
import os

import cv2
import numpy as np

from photo2fcstd.carve import carve, mesh_of
from photo2fcstd.errors import CaptureError

BOP_ROOT = os.environ.get("P2F_BOP", os.path.expanduser("~/3DPrint/tools/data/bop"))
BOUND_MM = 90.0
DARK = 40


def scene_dir(dataset, split, obj_id):
    for base in (os.path.join(BOP_ROOT, dataset, split), os.path.join(BOP_ROOT, split)):
        path = os.path.join(base, "%06d" % obj_id)
        if os.path.isdir(path):
            return path
    raise CaptureError("no scene for object %d under %s - is the archive unpacked?" % (obj_id, BOP_ROOT))


def load_scene(path):
    cameras = json.load(open(os.path.join(path, "scene_camera.json")))
    poses = json.load(open(os.path.join(path, "scene_gt.json")))
    return cameras, poses


def views_of(path, limit=None, stride=1):
    cameras, poses = load_scene(path)
    keys = sorted(cameras, key=int)[::stride]
    keys = keys[:limit] if limit else keys
    out = []
    for key in keys:
        gt = poses[key][0]
        R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"], float).reshape(3, 1)
        K = np.array(cameras[key]["cam_K"], float).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
        image = os.path.join(path, "rgb", "%s.png" % key.zfill(6))
        if not os.path.exists(image):
            image = os.path.join(path, "rgb", "%s.jpg" % key.zfill(6))
        out.append({"rvec": rvec, "tvec": t, "K": K, "dist": np.zeros(5), "image": image, "key": key})
    return out


def object_mask(path_or_image, provided=None):
    if provided and os.path.exists(provided):
        return cv2.imread(provided, cv2.IMREAD_GRAYSCALE) > 127
    image = cv2.imread(path_or_image) if isinstance(path_or_image, str) else path_or_image
    if image is None:
        raise CaptureError("cannot read %s" % path_or_image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = grey > DARK
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def masks_for(path, views):
    out = []
    for v in views:
        provided = os.path.join(path, "mask", "%s_000000.png" % v["key"].zfill(6))
        out.append(object_mask(v["image"], provided))
    return out


def truth_mesh(obj_id, kind="models_cad"):
    import trimesh
    for base in (os.path.join(BOP_ROOT, kind), os.path.join(BOP_ROOT, "tless", kind)):
        path = os.path.join(base, "obj_%06d.ply" % obj_id)
        if os.path.exists(path):
            return trimesh.load(path)
    raise CaptureError("no CAD model for object %d under %s" % (obj_id, BOP_ROOT))


def carve_object(obj_id, dataset="tless", split="train_primesense", views=24, voxel_mm=1.0,
                 bound_mm=BOUND_MM, allow_misses=1):
    path = scene_dir(dataset, split, obj_id)
    every = views_of(path)
    stride = max(1, len(every) // views)
    chosen = every[::stride][:views]
    bounds = ((-bound_mm, bound_mm), (-bound_mm, bound_mm), (-bound_mm, bound_mm))
    carved = carve(chosen, masks_for(path, chosen), voxel_mm=voxel_mm, allow_misses=allow_misses, bounds=bounds)
    if carved is None:
        raise CaptureError("nothing survived carving for object %d" % obj_id)
    carved["views_used"] = len(chosen)
    carved["object"] = obj_id
    return carved


def compare_to_truth(carved, obj_id):
    truth = truth_mesh(obj_id)
    got = np.sort(carved["extents_mm"])
    want = np.sort(truth.extents)
    return {"object": obj_id, "views": carved["views_used"], "voxel_mm": carved["voxel_mm"],
            "extents_mm": [round(float(x), 2) for x in got],
            "truth_mm": [round(float(x), 2) for x in want],
            "error_mm": [round(float(a - b), 2) for a, b in zip(got, want)],
            "worst_mm": round(float(np.max(np.abs(got - want))), 2),
            "volume_ratio": round(float(carved["volume_mm3"] / max(truth.volume, 1e-9)), 3)}
