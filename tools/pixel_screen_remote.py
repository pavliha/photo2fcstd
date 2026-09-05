"""Does light beat the contour? Contour-only vs contour+normal-intensity-profile, on T-LESS.

Runs entirely on a rented box: builds the corpus from tless/train_primesense (real photographs,
public poses), labels through the pose via the production tracer on the clean CAD silhouette,
samples seven grey intensities along each contour point's normal, then trains two arms of the
same architecture - the contour-only arm has the intensity channels zeroed - split by object.
The one axis no prior attempt changed is the input's information content; this changes it.
"""
import glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/src")
os.environ.setdefault("P2F_SEQ_DEVICE", "cuda")

N_POINTS = 192
PROF = 7
OFFS = np.linspace(-6, 6, PROF)


def resample(contour, n=N_POINTS):
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, np.concatenate([contour[:, 0], contour[:1, 0]])),
                            np.interp(t, s, np.concatenate([contour[:, 1], contour[:1, 1]]))])


def normal_profile(gray, pts):
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nor = np.column_stack([-t[:, 1], t[:, 0]])
    H, W = gray.shape
    out = np.zeros((len(pts), PROF), np.float32)
    for j, off in enumerate(OFFS):
        q = pts + nor * off
        out[:, j] = gray[np.clip(q[:, 1], 0, H - 1).astype(int), np.clip(q[:, 0], 0, W - 1).astype(int)]
    return (out - out.mean()) / (out.std() + 1e-6)


def silhouette(mesh, K, R, t, shape):
    V = np.asarray(mesh.vertices, float)
    cam = (R @ V.T).T + t
    z = np.clip(cam[:, 2], 1e-6, None)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    img = np.zeros(shape, np.uint8)
    for f in np.stack([u, v], axis=1)[np.asarray(mesh.faces)].astype(np.int32):
        cv2.fillConvexPoly(img, f, 1)
    return img > 0


def build():
    import trimesh
    from photo2fcstd.trace import elements
    rows = []
    rng = np.random.default_rng(0)
    for sc in sorted(glob.glob("/workspace/tless/train_primesense/*")):
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        gi = json.load(open(os.path.join(sc, "scene_gt_info.json")))
        cams = json.load(open(os.path.join(sc, "scene_camera.json")))
        obj = gt[list(gt)[0]][0]["obj_id"]
        mesh = trimesh.load("/workspace/tless/models_cad/obj_%06d.ply" % obj)
        ids = [i for i in gi if gi[i][0].get("visib_fract", 0) > 0.97]
        ids = [ids[i] for i in rng.choice(len(ids), min(400, len(ids)), replace=False)]
        for im_id in ids:
            mask = cv2.imread(os.path.join(sc, "mask_visib", "%06d_%06d.png" % (int(im_id), 0)), 0)
            img = cv2.imread(os.path.join(sc, "rgb", "%06d.png" % int(im_id)), 0)
            if mask is None or img is None:
                continue
            mask = mask > 128
            K = np.array(cams[im_id]["cam_K"], float).reshape(3, 3)
            g = gt[im_id][0]
            sil = silhouette(mesh, K, np.array(g["cam_R_m2c"], float).reshape(3, 3),
                             np.array(g["cam_t_m2c"], float), mask.shape)
            cr, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cc, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cr or not cc:
                continue
            real = max(cr, key=cv2.contourArea).reshape(-1, 2).astype(float)
            clean = max(cc, key=cv2.contourArea).reshape(-1, 2).astype(float)
            if len(real) < 80 or len(clean) < 80:
                continue
            length = float(max(np.ptp(clean[:, 0]), np.ptp(clean[:, 1])))
            try:
                els = elements(np.asarray(clean, float), length)
            except Exception:
                continue
            spans = [(np.asarray(e["p1"], float), 1 if e["type"] in ("arc", "bsplinecurve") else 0)
                     for e in els]
            if not (2 <= len(spans) <= 40):
                continue
            pts = resample(real)
            prof = normal_profile(img.astype(np.float32), pts)
            idxs, ok = [], True
            for tail, typ in spans:
                d = np.linalg.norm(pts - tail, axis=1)
                if d.min() > 12:
                    ok = False
                    break
                idxs.append((int(d.argmin()), typ))
            if not ok:
                continue
            dedup = []
            for e, typ in sorted(set(idxs)):
                if not dedup or e - dedup[-1][0] >= 2:
                    dedup.append((e, typ))
            if len(dedup) < 2:
                continue
            r = int(rng.integers(N_POINTS))
            pts = np.roll(pts, r, axis=0)
            prof = np.roll(prof, r, axis=0)
            sp = sorted(((e + r) % N_POINTS, typ) for e, typ in dedup)
            th_ = rng.uniform(0, 2 * np.pi)
            Rm = np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]])
            q = (pts - pts.mean(axis=0)) @ Rm.T
            scale = float(np.abs(q).max()) or 1.0
            rows.append((obj, (q / scale).astype(np.float32), prof, sp))
    print("built %d samples over %d objects" % (len(rows), len({r[0] for r in rows})), flush=True)
    return rows


def train_arm(rows, use_pixels, tag, seed=0):
    import torch
    import torch.nn as nn
    from photo2fcstd import seqnet
    X = np.stack([np.concatenate([seqnet.features(x[None])[0], p], axis=-1) for _, x, p, _ in rows])
    if not use_pixels:
        X = X.copy()
        X[..., 5:] = 0.0
    G = np.array([o for o, _, _, _ in rows])
    T = np.full((len(rows), seqnet.MAX_LEN), -100, np.int64)
    for i, (_, _, _, sp) in enumerate(rows):
        toks = [seqnet.TOK_LINE]
        for e, t in sp:
            toks += [seqnet.TOK_ARC if t == 1 else seqnet.TOK_LINE, e]
        toks.append(seqnet.TOK_EOS)
        T[i, :len(toks)] = toks[:seqnet.MAX_LEN]
    uniq = np.unique(G)
    rng = np.random.default_rng(seed)
    val_obj = set(uniq[rng.choice(len(uniq), 6, replace=False)])
    va = np.array([g in val_obj for g in G])
    model = seqnet.SeqNet(in_dim=5 + PROF).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    Xt, Tt = X[~va], T[~va]
    best = 0.0
    for ep in range(60):
        order = np.random.permutation(len(Xt))
        for i in range(0, len(order), 256):
            idx = order[i:i + 256]
            x = torch.from_numpy(Xt[idx]).cuda()
            t = torch.from_numpy(Tt[idx]).cuda()
            loss = loss_fn(model(x, t[:, :-1].clamp_min(0)).reshape(-1, seqnet.VOCAB),
                           t[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); ok = n = 0
        with torch.no_grad():
            Xv, Tv = X[va], T[va]
            for i in range(0, len(Xv), 512):
                x = torch.from_numpy(Xv[i:i + 512]).cuda()
                t = torch.from_numpy(Tv[i:i + 512]).cuda()
                lg = model(x, t[:, :-1].clamp_min(0))
                keep = t[:, 1:] != -100
                ok += int((lg.argmax(-1)[keep] == t[:, 1:][keep]).sum()); n += int(keep.sum())
        model.train()
        best = max(best, ok / max(n, 1))
    print("ARM %s seed %d: held-out-object token acc %.4f" % (tag, seed, best), flush=True)
    return best


if __name__ == "__main__":
    rows = build()
    results = {"contour": [], "pixels": []}
    for seed in (0, 1):
        results["contour"].append(train_arm(rows, False, "contour-only", seed))
        results["pixels"].append(train_arm(rows, True, "contour+pixels", seed))
    import numpy as np
    print("SCREENRESULT contour %.4f pixels %.4f delta %+.4f"
          % (np.mean(results["contour"]), np.mean(results["pixels"]),
             np.mean(results["pixels"]) - np.mean(results["contour"])), flush=True)
