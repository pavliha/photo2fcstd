"""Can camera poses come from the scene instead of a printed board?

Carving reaches 0.782 solid IoU on real photographs against the photo pipeline's ~0.43, and the only
thing it needs that an ordinary capture lacks is where each camera was. Today that comes from a
ChArUco board. Structure-from-motion gets it from the scene itself - which is how casual
photogrammetry works - and needs nothing printed.

This is the first gate, and it is synthetic on purpose: renders with a textured ground plane and
exactly known poses, so the question is only whether SfM plus carving reproduces carving with true
poses. If it fails here it fails everywhere. If it passes, what remains untested is whether a real
desk carries enough texture, and that needs real photographs - but only then.

A textureless part cannot register on its own, so the ground plane is what SfM sees. That is the
same bargain a real capture makes.
"""
import json, os, shutil, subprocess, sys, tempfile

import cv2
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import carve as C  # noqa: E402

W = H = 900
F = 1200.0
COLMAP = shutil.which("colmap") or "/opt/homebrew/bin/colmap"


def K_of():
    return np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def poses(n, radius, elevations=(22.0, 38.0, 55.0)):
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = np.radians(elevations[i % len(elevations)])
        eye = np.array([radius * np.cos(el) * np.cos(az), radius * np.cos(el) * np.sin(az),
                        radius * np.sin(el)])
        f = -eye / np.linalg.norm(eye)
        up = np.array([0, 0, 1.0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        R = np.stack([r, d, f])
        assert abs(np.linalg.det(R) - 1) < 1e-6
        rvec, _ = cv2.Rodrigues(R)
        out.append({"rvec": rvec, "tvec": (-R @ eye).reshape(3, 1), "K": K_of(),
                    "dist": np.zeros(5), "R": R, "eye": eye})
    return out


def ground(texture, view, extent):
    """The z=0 plane, warped exactly through its own homography, so SfM sees real geometry."""
    R, t = view["R"], view["tvec"].reshape(3)
    Hm = view["K"] @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    src = np.float32([[0, 0], [texture.shape[1], 0], [texture.shape[1], texture.shape[0]], [0, texture.shape[0]]])
    dst = np.float32([[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]])
    plane = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(texture, Hm @ plane, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))


def render(mesh, view, texture, extent, rng):
    img = ground(texture, view, extent)
    V = np.asarray(mesh.vertices, float)
    Faces = np.asarray(mesh.faces)
    uv = C.project(V, view)
    cam = (view["R"] @ V.T + view["tvec"]).T
    depth = cam[Faces].mean(1)[:, 2]
    fn = np.asarray(mesh.face_normals, float) @ view["R"].T
    fn = fn * np.sign(np.where(fn[:, 2:3] == 0, 1, -fn[:, 2:3]))
    light = np.array([0.3, -0.4, -0.9])
    light /= np.linalg.norm(light)
    shade = 45 + 190 * np.clip(fn @ (-light), 0, 1)
    mask = np.zeros((H, W), np.uint8)
    tri = np.round(uv[Faces]).astype(np.int32)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, tri[i], (float(shade[i]),) * 3)
        cv2.fillConvexPoly(mask, tri[i], 1)
    img = np.clip(img.astype(float) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img, mask > 0


def run_colmap(image_dir, work, camera_model="PINHOLE"):
    db = os.path.join(work, "db.db")
    sparse = os.path.join(work, "sparse")
    os.makedirs(sparse, exist_ok=True)
    steps = [
        [COLMAP, "feature_extractor", "--database_path", db, "--image_path", image_dir,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera_model,
         "--FeatureExtraction.use_gpu", "0"],
        [COLMAP, "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "0"],
        [COLMAP, "mapper", "--database_path", db, "--image_path", image_dir, "--output_path", sparse],
    ]
    for cmd in steps:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if p.returncode != 0:
            return None, "%s failed: %s" % (cmd[1], (p.stderr or p.stdout)[-300:])
    models = sorted(glob_dirs(sparse))
    if not models:
        return None, "mapper produced no model"
    return os.path.join(sparse, models[-1]), None


def glob_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def read_images(model_dir):
    """COLMAP images.txt: image_id qw qx qy qz tx ty tz camera_id name."""
    out = {}
    txt = os.path.join(model_dir, "images.txt")
    if not os.path.exists(txt):
        subprocess.run([COLMAP, "model_converter", "--input_path", model_dir,
                        "--output_path", model_dir, "--output_type", "TXT"],
                       capture_output=True, text=True)
    with open(txt) as fh:
        lines = [l for l in fh if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        f = lines[i].split()
        q = np.array([float(x) for x in f[1:5]])
        t = np.array([float(x) for x in f[5:8]])
        out[f[9]] = (quat_to_R(q), t)
    return out


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src, dst):
    """The similarity that takes SfM's arbitrary frame to the metric one, from camera centres."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd((D.T @ S) / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    scale = float(np.trace(np.diag(sig) @ E) / max((S ** 2).sum() / len(src), 1e-12))
    return scale, R, mu_d - scale * R @ mu_s


def main(part=None, views=24, voxel=0.6, contrast=1.0, quiet=False):
    from photo2fcstd import bench, sketch_score as SS
    rng = np.random.default_rng(0)
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    part = part or next(p for p in sorted(ideal) if SS.trustworthy(ideal[p]))
    mesh = trimesh.load(bench.truth_of(part))
    mesh.apply_translation(-np.array([mesh.bounds[:, 0].mean(), mesh.bounds[:, 1].mean(), mesh.bounds[0][2]]))
    radius = float(np.max(mesh.extents)) * 1.9
    truth = poses(views, radius)
    lo, hi = 128 - 98 * contrast, 128 + 97 * contrast
    texture = rng.integers(int(lo), int(hi) + 1, (600, 600, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.4)

    work = tempfile.mkdtemp(prefix="sfm_")
    images = os.path.join(work, "images")
    os.makedirs(images, exist_ok=True)
    masks = []
    for i, v in enumerate(truth):
        img, m = render(mesh, v, texture, float(np.max(mesh.extents)) * 1.6, rng)
        cv2.imwrite(os.path.join(images, "v%03d.png" % i), img)
        masks.append(m)
    if not quiet:
        print("  %s: %d views rendered, part covers %.1f%% of frame\n"
              % (part, views, 100 * np.mean([m.mean() for m in masks])))

    model, err = run_colmap(images, work)
    if model is None:
        shutil.rmtree(work, ignore_errors=True)
        if quiet:
            return {"part": part, "views": views, "contrast": contrast, "registered": 0}
        print("  SfM failed: %s" % err)
        return
    got = read_images(model)
    if not quiet:
        print("  SfM registered %d of %d images" % (len(got), views))
    if len(got) < 8:
        shutil.rmtree(work, ignore_errors=True)
        if quiet:
            return {"part": part, "views": views, "contrast": contrast, "registered": len(got)}
        print("  too few to carve from")
        return

    names = sorted(got)
    idx = [int(n[1:4]) for n in names]
    sfm_c = np.array([-got[n][0].T @ got[n][1] for n in names])
    true_c = np.array([truth[i]["eye"] for i in idx])
    scale, Rw, tw = umeyama(sfm_c, true_c)
    aligned = (scale * (Rw @ sfm_c.T).T) + tw
    centre_err = np.linalg.norm(aligned - true_c, axis=1)
    rot_err = []
    for n, i in zip(names, idx):
        R_est = got[n][0] @ Rw.T
        d = R_est @ truth[i]["R"].T
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    if not quiet:
        print("  camera centre error: median %.2f mm of a %.0f mm radius (%.2f%%)"
              % (np.median(centre_err), radius, 100 * np.median(centre_err) / radius))
        print("  camera rotation error: median %.3f deg, worst %.3f deg\n"
              % (np.median(rot_err), np.max(rot_err)))

    est_views = []
    for n, i in zip(names, idx):
        R_est = got[n][0] @ Rw.T
        c_est = aligned[names.index(n)]
        est_views.append({"rvec": cv2.Rodrigues(R_est)[0], "tvec": (-R_est @ c_est).reshape(3, 1),
                          "K": K_of(), "dist": np.zeros(5)})
    bounds = [(float(mesh.bounds[0][k]) - 2, float(mesh.bounds[1][k]) + 2) for k in range(2)] \
        + [(0.0, float(mesh.bounds[1][2]) + 2)]
    chosen = [masks[i] for i in idx]
    a = C.carve([truth[i] for i in idx], chosen, voxel_mm=voxel, bounds=bounds)
    b = C.carve(est_views, chosen, voxel_mm=voxel, bounds=bounds)
    if a is None or b is None:
        shutil.rmtree(work, ignore_errors=True)
        if quiet:
            return {"part": part, "views": views, "contrast": contrast,
                    "registered": len(got), "rot_deg": float(np.median(rot_err)), "agree": 0.0}
        print("  carving produced nothing (%s / %s)" % (a is None, b is None))
        return
    A = set(map(tuple, np.round(a["points_mm"] / voxel).astype(int)))
    B = set(map(tuple, np.round(b["points_mm"] / voxel).astype(int)))
    agree = len(A & B) / max(len(A | B), 1)
    shutil.rmtree(work, ignore_errors=True)
    if quiet:
        return {"part": part, "views": views, "contrast": contrast, "registered": len(got),
                "rot_deg": float(np.median(rot_err)), "agree": float(agree)}
    print("  %-34s %s" % ("carved from true poses", len(A)))
    print("  %-34s %s" % ("carved from SfM poses", len(B)))
    print("  %-34s %.3f" % ("agreement between the two", agree))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [None]))
