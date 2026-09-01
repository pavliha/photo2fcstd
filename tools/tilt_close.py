"""Does knowing the tilt actually improve the drawing?

The pixel probe recovers a part's face normal to 4.8 degrees on T-LESS, but a good angle is not
the objective - the sketch is. This closes the loop inside PrintCAD, where a real ideal sketch
exists to score against: render a part shaded at a known tilt, predict the normal from the render
held out by part, rectify the silhouette with the predicted normal, trace it and score it.

Three arms on identical views:

- `as shot`     what the pipeline does now, no rectification
- `predicted`   rectified by the model's normal - what shipping this would buy
- `true normal` rectified by the real normal - the ceiling, and the check that the rectifier works

If `true normal` does not clearly beat `as shot`, the rectifier is broken and the middle arm means
nothing; that check comes first.
"""
import json, os, sys

import cv2
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import carve as C, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

W = H = 640
F = 1000.0
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def K_of():
    return np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def face_normal(record):
    if not record.get("loops"):
        return None
    pts = np.array([e[k] for loop in record["loops"] for e in loop
                    for k in ("c", "p0", "p1") if k in e], float)
    if len(pts) < 3:
        return None
    q = pts - pts.mean(0)
    u, s, vt = np.linalg.svd(q, full_matrices=False)
    return vt[-1] if s[-1] / max(s[0], 1e-9) < 1e-3 else None


def view_at(normal, tilt_deg, azimuth_deg, radius):
    """A camera looking at the origin, tilted `tilt_deg` off the face normal."""
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    t, az = np.radians(tilt_deg), np.radians(azimuth_deg)
    eye = radius * (np.cos(t) * n + np.sin(t) * (np.cos(az) * e1 + np.sin(az) * e2))
    f = -eye / np.linalg.norm(eye)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.95 else np.array([0, 1.0, 0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    assert abs(np.linalg.det(R) - 1) < 1e-6
    rvec, _ = cv2.Rodrigues(R)
    return {"rvec": rvec, "tvec": (-R @ eye).reshape(3, 1), "K": K_of(), "dist": np.zeros(5), "R": R}


def shaded(mesh, view):
    """Lambertian, painter's algorithm. Enough shading to carry a normal, no z-buffer needed."""
    V = np.asarray(mesh.vertices, float)
    Faces = np.asarray(mesh.faces)
    uv = C.project(V, view)
    R = view["R"]
    cam = (R @ V.T + view["tvec"]).T
    depth = cam[Faces].mean(1)[:, 2]
    fn = np.asarray(mesh.face_normals, float) @ R.T
    light = np.array([0.3, -0.5, -0.8])
    light /= np.linalg.norm(light)
    shade = 40 + 200 * np.clip(np.abs(fn @ light), 0, 1)
    img = np.zeros((H, W), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    tri = np.round(uv[Faces]).astype(np.int32)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, tri[i], float(shade[i]))
        cv2.fillConvexPoly(mask, tri[i], 1)
    return img, mask > 0


def rectify(mask, normal_cam, K):
    """Map the face plane fronto-parallel: H = K R K^-1 with R taking the normal to the axis."""
    n = normal_cam / np.linalg.norm(normal_cam)
    n = n * np.sign(n[2] if n[2] != 0 else 1)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    R = np.eye(3) if s < 1e-9 else (
        np.eye(3) + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) ** 2 * ((1 - c) / s ** 2))
    Hm = K @ R @ np.linalg.inv(K)
    warped = cv2.warpPerspective(mask.astype(np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST)
    return warped > 0


def score_mask(mask, record):
    from photo2fcstd import analysis, spec as spec_mod
    view = analysis.view_from_mask(mask.astype(np.uint8))
    loops = spec_mod.traced_outline(view)
    if not loops:
        return None
    return SS.score_one({"outline": {"loops": loops}}, record)
