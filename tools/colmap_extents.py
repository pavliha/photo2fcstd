import json
import sys

import numpy as np


def read_points(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if not l.startswith("#")])


def read_cameras(path):
    rows = [l.split() for l in open(path) if not l.startswith("#")]
    rows = [r for r in rows if len(r) >= 10 and r[-1].endswith(".jpg")]
    cams = []
    for r in rows:
        qw, qx, qy, qz = (float(x) for x in r[1:5]); t = np.array([float(x) for x in r[5:8]])
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        cams.append(-R.T @ t)
    return np.array(cams)


def main(model_dir, out_json):
    P = read_points(model_dir + "/points3D.txt")
    c = np.median(P, 0)
    keep = np.linalg.norm(P - c, axis=1) < 3 * np.median(np.linalg.norm(P - c, axis=1))
    P = P[keep]
    axes = np.linalg.svd(P - P.mean(0), full_matrices=False)[2]
    q = (P - P.mean(0)) @ axes.T
    ext = np.quantile(q, 0.98, 0) - np.quantile(q, 0.02, 0)
    cams = read_cameras(model_dir + "/images.txt")
    dist = np.linalg.norm(cams - P.mean(0), axis=1)
    dirs = (cams - P.mean(0)) / dist[:, None]
    res = {"points": int(len(P)), "cameras": int(len(cams)), "long": float(ext[0]), "short": float(ext[1]), "height": float(ext[2]),
           "short_over_long": float(ext[1] / ext[0]), "height_over_long": float(ext[2] / ext[0]),
           "camera_dist_over_long": float(np.median(dist) / ext[0]), "camera_dist_spread": float(np.ptp(dist) / np.median(dist)),
           "view_arc_deg": float(np.degrees(np.arccos(np.clip(dirs @ dirs[0], -1, 1))).max())}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
