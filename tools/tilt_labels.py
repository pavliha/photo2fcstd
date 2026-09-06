import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def truth_vertices(part):
    loops = IDEAL[part]["loops"]
    if len(loops) != 1 or any(e["type"] != "line" for e in loops[0]):
        return None
    return np.array([np.asarray(e["xy"], float)[0] for e in loops[0]])


def match(O, T):
    def norm(P):
        P = P - P.mean(0)
        return P / np.ptp(P, 0).max()
    Tn = norm(T)
    best = (1e9, None)
    for flip in (1, -1):
        for rot in range(4):
            R = np.array([[np.cos(rot * np.pi / 2), -np.sin(rot * np.pi / 2)], [np.sin(rot * np.pi / 2), np.cos(rot * np.pi / 2)]])
            Or = norm((O * np.array([flip, 1])) @ R.T)
            for rev in (1, -1):
                for sh in range(len(O)):
                    d = np.abs(np.roll(Or[::rev], sh, axis=0) - Tn).mean()
                    if d < best[0]:
                        best = (d, (flip, rot, sh, rev))
    flip, rot, sh, rev = best[1]
    R = np.array([[np.cos(rot * np.pi / 2), -np.sin(rot * np.pi / 2)], [np.sin(rot * np.pi / 2), np.cos(rot * np.pi / 2)]])
    return np.roll(((O * np.array([flip, 1])) @ R.T)[::rev], sh, axis=0), best[0]


def raster(poly, box, size=400):
    mn, mx = box
    s = (size - 8) / max(np.ptp(np.vstack([mn, mx]), 0).max(), 1e-9)
    img = np.zeros((size, size), np.uint8)
    cv2.fillPoly(img, [((np.asarray(poly, float) - mn) * s + 4).astype(np.int32)], 1)
    return img > 0


def iou(a, b):
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def one(part):
    from photo2fcstd import bench, recognise
    T = truth_vertices(part)
    if T is None:
        return []
    out = []
    for i, p in enumerate(bench.photos_of(part)[:3]):
        try:
            spec, _ = recognise.route([p], name=part, rec={"single_extrusion": True, "face_photo_index": 0})
            loop = spec["outline"]["loops"][0]
            if loop.get("type") != "loop" or any(e["type"] != "line" for e in loop["elements"]):
                continue
            O = np.array([e["p0"] for e in loop["elements"]], float)
            if len(O) != len(T):
                continue
            Om, resid = match(O, T)
            if resid > 0.08:
                continue
            H, _ = cv2.findHomography(Om.astype(np.float32), T.astype(np.float32), 0)
            A, _ = cv2.estimateAffine2D(Om.astype(np.float32), T.astype(np.float32))
            sv = np.linalg.svd(A[:, :2], compute_uv=False)
            box = (T.min(0), T.max(0)); Tr = raster(T, box)
            Sim = (Om - Om.mean(0)) * (np.ptp(T, 0).max() / np.ptp(Om, 0).max()) + T.mean(0)
            Rect = cv2.perspectiveTransform(Om.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            out.append({"part": part, "view": i, "photo": p, "n_vertices": int(len(T)), "match_resid": float(resid),
                        "foreshorten": float(sv[1] / sv[0]), "tilt_deg": float(np.degrees(np.arccos(min(sv[1] / sv[0], 1.0)))),
                        "H": H.tolist(), "iou_similarity": iou(raster(Sim, box), Tr), "iou_rectified": iou(raster(Rect, box), Tr)})
        except Exception:
            continue
    return out


def main(n=150):
    from photo2fcstd import bench, sketch_score as SS
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0]
    rng = np.random.default_rng(0)
    parts = [parts[i] for i in rng.choice(len(parts), min(n, len(parts)), replace=False)]
    with Pool(6) as pool:
        rows = [r for rs in pool.map(one, parts) for r in rs]
    out = os.path.join(ROOT, "runs", "tilt_labels.json")
    json.dump(rows, open(out, "w"), indent=1)
    if rows:
        sim = np.array([r["iou_similarity"] for r in rows]); rec = np.array([r["iou_rectified"] for r in rows]); tilt = np.array([r["tilt_deg"] for r in rows])
        print("TILTLABELS parts=%d photos_labelled=%d | tilt median %.1f deg (p10 %.1f, p90 %.1f) | IoU similarity %.3f -> rectified %.3f (+%.3f) | rectified >= 0.9 on %.0f%%"
              % (n, len(rows), np.median(tilt), np.quantile(tilt, .1), np.quantile(tilt, .9), sim.mean(), rec.mean(), rec.mean() - sim.mean(), 100 * (rec >= 0.9).mean()))
    else:
        print("TILTLABELS none")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
