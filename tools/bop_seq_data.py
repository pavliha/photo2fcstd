"""The sixteenth attempt's corpus: real BOP contours labelled through the pose.

Input: the real modal-mask contour of a near-unoccluded instance - the deployment distribution
itself, unsimulated. Labels: the production tracer applied to the *clean* CAD silhouette projected
through the given pose, its breakpoints and span types transferred to the real contour across the
measured 1.4-1.9 px boundary gap. Teacher on clean input, student on the paired real input - the
conservation law satisfied by construction for the first time. Balanced per object, split by
object downstream.
"""
import glob, json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bop_yield import BOP, load_mesh, scenes_of, silhouette  # noqa: E402
from seq_data import MAX_SPANS, N_POINTS, resample  # noqa: E402

PER_OBJECT = 500
MAX_GAP = 4.0


def pairs_of(ds, split="val"):
    out = []
    for sc in scenes_of(ds, split):
        cams = [""]
        if not os.path.exists(os.path.join(sc, "scene_gt_info.json")):
            cams = sorted(f[len("scene_gt_info_"):-len(".json")] for f in os.listdir(sc)
                          if f.startswith("scene_gt_info_") and f.endswith(".json"))
        for cam in cams:
            suf = ("_" + cam) if cam else ""
            gi = json.load(open(os.path.join(sc, "scene_gt_info%s.json" % suf)))
            gt = json.load(open(os.path.join(sc, "scene_gt%s.json" % suf)))
            for im_id, infos in gi.items():
                for k, info in enumerate(infos):
                    if info.get("visib_fract", 0) > 0.97:
                        out.append((ds, sc, suf, im_id, k, gt[im_id][k]["obj_id"]))
    return out


def tless_pairs():
    out = []
    root = os.path.join(BOP, "tless", "train_primesense")
    for sc in sorted(glob.glob(os.path.join(root, "*"))):
        gi = json.load(open(os.path.join(sc, "scene_gt_info.json")))
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        obj = gt[list(gt)[0]][0]["obj_id"]
        for im_id, infos in gi.items():
            if infos[0].get("visib_fract", 0) > 0.97:
                out.append(("tless", sc, "", im_id, 0, obj))
    return out


def teacher_spans(clean_contour, length_px):
    from photo2fcstd.trace import elements
    els = elements(np.asarray(clean_contour, float), length_px)
    spans, cursor_pts = [], []
    for e in els:
        run = e.get("_run")
        tail = np.asarray(e["p1"], float)
        spans.append((tail, 1 if e["type"] in ("arc", "bsplinecurve") else 0))
    return spans


def one(args):
    ds, sc, suf, im_id, k, obj = args
    try:
        mask_p = os.path.join(sc, "mask_visib%s" % suf, "%06d_%06d.png" % (int(im_id), k))
        mask = cv2.imread(mask_p, 0)
        if mask is None:
            return None
        mask = mask > 128
        cam = json.load(open(os.path.join(sc, "scene_camera%s.json" % suf)))[im_id]
        g = json.load(open(os.path.join(sc, "scene_gt%s.json" % suf)))[im_id][k]
        key = (ds, obj)
        if key not in MESHES:
            if ds == "tless":
                import trimesh
                MESHES[key] = trimesh.load(os.path.join(BOP, "tless", "models_cad", "obj_%06d.ply" % obj))
            else:
                MESHES[key] = load_mesh(ds, obj)
        mesh = MESHES[key]
        if mesh is None:
            return None
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        R = np.array(g["cam_R_m2c"], float).reshape(3, 3)
        t = np.array(g["cam_t_m2c"], float)
        sil = silhouette(mesh, K, R, t, mask.shape)
        creal, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cclean, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not creal or not cclean:
            return None
        real = max(creal, key=cv2.contourArea).reshape(-1, 2).astype(float)
        clean = max(cclean, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(real) < 80 or len(clean) < 80:
            return None
        length = float(max(np.ptp(clean[:, 0]), np.ptp(clean[:, 1])))
        spans = teacher_spans(clean, length)
        if not (2 <= len(spans) <= MAX_SPANS):
            return None
        pts = resample(real, N_POINTS)
        idxs = []
        for tail, typ in spans:
            d = np.linalg.norm(pts - tail, axis=1)
            if d.min() > MAX_GAP * 3:
                return None
            idxs.append((int(d.argmin()), typ))
        idxs = sorted(set(idxs))
        dedup = []
        for e, typ in idxs:
            if not dedup or e - dedup[-1][0] >= 2:
                dedup.append((e, typ))
        if len(dedup) < 2:
            return None
        rng = np.random.default_rng(abs(hash((sc, im_id, k))) % (1 << 31))
        r = int(rng.integers(N_POINTS))
        pts = np.roll(pts, r, axis=0)
        sp = sorted(((e + r) % N_POINTS, typ) for e, typ in dedup)
        th_ = rng.uniform(0, 2 * np.pi)
        Rm = np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]])
        q = (pts - pts.mean(axis=0)) @ Rm.T
        scale = float(np.abs(q).max()) or 1.0
        return ("%s/%d" % (ds, obj), (q / scale).astype(np.float32), sp)
    except Exception:
        return None


MESHES = {}


def build(out="data/bop_seq.npz", seed=0):
    rng = np.random.default_rng(seed)
    allp = []
    for ds in ("itodd", "ipd", "xyzibd"):
        allp += pairs_of(ds)
    allp += tless_pairs()
    by_obj = {}
    for p in allp:
        by_obj.setdefault((p[0], p[5]), []).append(p)
    picked = []
    for key, lst in by_obj.items():
        idx = rng.choice(len(lst), min(PER_OBJECT, len(lst)), replace=False)
        picked += [lst[i] for i in idx]
    print("objects %d, picked %d instance pairs" % (len(by_obj), len(picked)))
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, picked) if r is not None]
    X = np.stack([x for _, x, _ in rows])
    G = np.array([g for g, _, _ in rows])
    flat = [[v for e, t in sp for v in (int(t), int(e))] + [-1] for _, _, sp in rows]
    L = max(len(f) for f in flat)
    seqs = np.full((len(flat), L), -2, np.int32)
    for i, f in enumerate(flat):
        seqs[i, :len(f)] = f
    np.savez_compressed(os.path.join(ROOT, out), X=X, seq=seqs, group=G)
    curved = np.mean([t for _, _, sp in rows for _, t in sp])
    print("%d real labelled contours (%.0f%% of picked), %d objects, %.0f%% curved spans"
          % (len(rows), 100 * len(rows) / max(len(picked), 1), len(set(G)), 100 * curved))


if __name__ == "__main__":
    build(*(sys.argv[1:2] or []))
