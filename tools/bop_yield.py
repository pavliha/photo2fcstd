"""The label factory's yield on BOP-Industrial: how many real photographs label themselves.

For every ground-truth instance in the validation splits (the splits with public poses): count
instances essentially unoccluded (visib_fract), and on a sample, measure how far the dataset's
real modal mask boundary sits from the CAD model's silhouette projected through the given pose -
the pose-exact analogue of label_screen's 2.6 px alignment error.
"""
import glob, json, os, sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOP = os.path.join(ROOT, "data", "bop")


def scenes_of(ds, split):
    for pat in ((ds, ds, split), (ds, split), (ds, ds + "_" + split, split)):
        got = sorted(glob.glob(os.path.join(BOP, *pat, "*")))
        if got:
            return got
    return []


def load_mesh(ds, obj_id):
    import trimesh
    for sub in (os.path.join(BOP, ds, ds), os.path.join(BOP, ds)):
        for mdir in ("models", "models_eval"):
            p = os.path.join(sub, mdir, "obj_%06d.ply" % obj_id)
            if os.path.exists(p):
                return trimesh.load(p)
    return None


def silhouette(mesh, K, R, t, shape):
    V = np.asarray(mesh.vertices, float)
    cam = (R @ V.T).T + t
    z = np.clip(cam[:, 2], 1e-6, None)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    img = np.zeros(shape, np.uint8)
    tri = np.stack([u, v], axis=1)[np.asarray(mesh.faces)].astype(np.int32)
    for f in tri:
        cv2.fillConvexPoly(img, f, 1)
    return img > 0


def boundary_dist(mask_a, mask_b):
    ca, _ = cv2.findContours(mask_a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cb, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ca or not cb:
        return None
    pa = max(ca, key=cv2.contourArea).reshape(-1, 2).astype(float)
    pb = max(cb, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(pa) < 20 or len(pb) < 20:
        return None
    d = np.sqrt(((pa[::5, None] - pb[None, ::5]) ** 2).sum(-1)).min(1)
    return float(np.mean(d))


def one_dataset(ds, split, sample=50, seed=0):
    scenes = scenes_of(ds, split)
    total = hi95 = hi99 = 0
    objects = set()
    pairs = []
    for sc in scenes:
        cams = [""]
        if not os.path.exists(os.path.join(sc, "scene_gt_info.json")):
            cams = sorted(f[len("scene_gt_info_"):-len(".json")]
                          for f in os.listdir(sc)
                          if f.startswith("scene_gt_info_") and f.endswith(".json"))
        for cam in cams:
            suf = ("_" + cam) if cam else ""
            gi_path = os.path.join(sc, "scene_gt_info%s.json" % suf)
            gt_path = os.path.join(sc, "scene_gt%s.json" % suf)
            if not (os.path.exists(gi_path) and os.path.exists(gt_path)):
                continue
            gi = json.load(open(gi_path))
            gt = json.load(open(gt_path))
            for im_id, infos in gi.items():
                for k, info in enumerate(infos):
                    total += 1
                    vf = info.get("visib_fract", 0)
                    obj = gt[im_id][k]["obj_id"]
                    objects.add(obj)
                    if vf > 0.95:
                        hi95 += 1
                        pairs.append((sc, cam, im_id, k, obj))
                    if vf > 0.99:
                        hi99 += 1
    rng = np.random.default_rng(seed)
    dists = []
    meshes = {}
    if pairs:
        for sc, camname, im_id, k, obj in [pairs[i] for i in rng.choice(len(pairs), min(sample, len(pairs)), replace=False)]:
            suf = ("_" + camname) if camname else ""
            cam = json.load(open(os.path.join(sc, "scene_camera%s.json" % suf)))[im_id]
            gt = json.load(open(os.path.join(sc, "scene_gt%s.json" % suf)))[im_id][k]
            mask_p = os.path.join(sc, "mask_visib%s" % suf, "%06d_%06d.png" % (int(im_id), k))
            if not os.path.exists(mask_p):
                continue
            mask = cv2.imread(mask_p, 0) > 128
            if obj not in meshes:
                meshes[obj] = load_mesh(ds, obj)
            mesh = meshes[obj]
            if mesh is None:
                continue
            K = np.array(cam["cam_K"], float).reshape(3, 3)
            R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
            t = np.array(gt["cam_t_m2c"], float)
            sil = silhouette(mesh, K, R, t, mask.shape)
            d = boundary_dist(mask, sil)
            if d is not None:
                dists.append(d)
    return {"scenes": len(scenes), "instances": total, "objects": len(objects),
            "visib95": hi95, "visib99": hi99,
            "boundary_px": (float(np.median(dists)) if dists else None,
                            float(np.percentile(dists, 90)) if dists else None, len(dists))}


def main():
    rows = {}
    for ds, split in (("itodd", "val"), ("ipd", "val"), ("xyzibd", "val")):
        rows[ds] = one_dataset(ds, split)
    tless_root = os.path.join(BOP, "tless", "train_primesense")
    if os.path.isdir(tless_root):
        rows["tless-train"] = one_dataset_tless(tless_root)
    print("%-12s %7s %9s %8s %10s %10s   %s" % ("dataset", "scenes", "instances", "objects",
                                                "visib>95%", "visib>99%", "mask-vs-pose boundary (median/p90 px, n)"))
    for ds, r in rows.items():
        b = r["boundary_px"]
        print("%-12s %7d %9d %8d %10d %10d   %s" % (ds, r["scenes"], r["instances"], r["objects"],
              r["visib95"], r["visib99"],
              "%.1f / %.1f px (n=%d)" % b if b[0] is not None else "no sample"))


def one_dataset_tless(root):
    global BOP
    scenes = sorted(glob.glob(os.path.join(root, "*")))
    total = hi95 = hi99 = 0
    objects = set()
    dists = []
    import trimesh
    rng = np.random.default_rng(0)
    sel = rng.choice(len(scenes), min(6, len(scenes)), replace=False)
    for si, sc in enumerate(scenes):
        gi = json.load(open(os.path.join(sc, "scene_gt_info.json")))
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        cams = json.load(open(os.path.join(sc, "scene_camera.json")))
        obj = gt[list(gt)[0]][0]["obj_id"]
        objects.add(obj)
        n = len(gi)
        total += n
        hi95 += sum(1 for infos in gi.values() if infos[0].get("visib_fract", 0) > 0.95)
        hi99 += sum(1 for infos in gi.values() if infos[0].get("visib_fract", 0) > 0.99)
        if si in sel:
            mesh = trimesh.load(os.path.join(BOP, "tless", "models_cad", "obj_%06d.ply" % obj))
            for im_id in list(gi)[::200]:
                mask_p = os.path.join(sc, "mask_visib", "%06d_%06d.png" % (int(im_id), 0))
                if not os.path.exists(mask_p):
                    continue
                mask = cv2.imread(mask_p, 0) > 128
                K = np.array(cams[im_id]["cam_K"], float).reshape(3, 3)
                g = gt[im_id][0]
                sil = silhouette(mesh, K, np.array(g["cam_R_m2c"], float).reshape(3, 3),
                                 np.array(g["cam_t_m2c"], float), mask.shape)
                d = boundary_dist(mask, sil)
                if d is not None:
                    dists.append(d)
    return {"scenes": len(scenes), "instances": total, "objects": len(objects),
            "visib95": hi95, "visib99": hi99,
            "boundary_px": (float(np.median(dists)) if dists else None,
                            float(np.percentile(dists, 90)) if dists else None, len(dists))}


if __name__ == "__main__":
    main()
