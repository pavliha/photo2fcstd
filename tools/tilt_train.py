import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def affine_target(H):
    H = np.asarray(H, float); H = H / H[2, 2]
    A = H[:2, :2]
    return (A / np.linalg.svd(A, compute_uv=False)[0]).ravel()


def apply_affine(Om, a, T, M=None):
    if len(a) == 3:                                  # symmetric stretch in the image frame -> bring into Om's frame
        P = np.array([[a[0], a[1]], [a[1], a[2]]]); M = np.eye(2) if M is None else np.asarray(M)
        A = M @ P @ np.linalg.inv(M)
    else:
        A = a.reshape(2, 2)
    Q = (Om - Om.mean(0)) @ A.T
    Q = Q * (np.ptp(T, 0).max() / max(np.ptp(Q, 0).max(), 1e-9))
    return Q - Q.mean(0) + T.mean(0)


def main(labels_path):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from photo2fcstd import recognise, tilt_model
    from tilt_labels import truth_vertices, match, raster, iou
    rows = json.load(open(labels_path))
    X, Y, meta = [], [], []
    for r in rows:
        v = tilt_model.vector_of(r["photo"])
        if v is None:
            continue
        X.append(v); Y.append(np.asarray(r["P_img"], float)[np.triu_indices(2)] if "P_img" in r else affine_target(r["H"])); meta.append(r)
    X = np.array(X); Y = np.array(Y); groups = np.array([m["part"] for m in meta])
    print("DATA photos=%d parts=%d dim=%d" % (len(X), len(set(groups)), X.shape[1]))
    n_splits = min(5, len(set(groups)))
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, Y, groups):
        m = Ridge(alpha=float(os.environ.get("TILT_ALPHA", "10.0"))).fit(X[tr], Y[tr]); pred[te] = m.predict(X[te])
    mean_pred = np.tile(Y.mean(0), (len(Y), 1))
    res = {"similarity": [], "mean_affine": [], "predicted_affine": [], "oracle_affine": [], "oracle_H": []}
    for k, r in enumerate(meta):
        T = truth_vertices(r["part"])
        spec, _ = recognise.route([r["photo"]], name=r["part"], rec={"single_extrusion": True, "face_photo_index": 0})
        O = np.array([e["p0"] for e in spec["outline"]["loops"][0]["elements"]], float)
        Om, _ = match(O, T); box = (T.min(0), T.max(0)); Tr = raster(T, box)
        Mk = r.get("M")
        res["similarity"].append(iou(raster(apply_affine(Om, np.eye(2).ravel(), T), box), Tr))
        res["mean_affine"].append(iou(raster(apply_affine(Om, mean_pred[k], T, Mk), box), Tr))
        res["predicted_affine"].append(iou(raster(apply_affine(Om, pred[k], T, Mk), box), Tr))
        res["oracle_affine"].append(iou(raster(apply_affine(Om, Y[k], T, Mk), box), Tr))
        H = np.asarray(r["H"], np.float32)
        res["oracle_H"].append(iou(raster(cv2.perspectiveTransform(Om.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2), box), Tr))
    line = " | ".join("%s %.3f" % (k, np.mean(v)) for k, v in res.items())
    print("TILTHEAD out-of-fold by part: " + line)
    m = Ridge(alpha=float(os.environ.get("TILT_ALPHA", "10.0"))).fit(X, Y)
    import joblib
    joblib.dump({"model": m, "mean": Y.mean(0), "n": len(X)}, os.path.join(ROOT, "data", "tilt_affine.joblib"))
    json.dump({k: [float(x) for x in v] for k, v in res.items()}, open(os.path.join(ROOT, "runs", "tilt_head_eval.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "tilt_labels.json"))
