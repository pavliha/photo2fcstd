"""Depth measured from photographs alone: COLMAP poses, desk plane from the scene, carve.

The archive's three photographs do not contain depth (tools/depth_consistency.py). Sixteen
photographs of the part on a textured surface do: SfM recovers the cameras from the scene, the
desk is the dominant plane in the sparse cloud, and carving measures the part - depth included -
in the desk's frame. Scale stays unknown until one caliper reading; everything else is measured.

Gated on rendered frames through the tool's own full path - COLMAP with estimated intrinsics,
desk plane from the sparse cloud, no truth consulted: depth/length 1.0% off at 3.73 mm thickness,
and a +0.5 to +0.8 mm absolute overestimate on parts of 2.5 mm and thinner, which is the visual
hull's own recorded accuracy (carve_check: +0.71/+0.35/+0.85 mm with exact poses) - SfM adds
nothing on top. A steeper 70 deg camera ring makes it worse, not better: the low views are what
carve the sides. Depth is read as the median top-surface height over interior columns, because the
hull cannot carve the skirt at the base rim and any whole-extent measure inherits it.

Usage: python tools/sfm_real.py <image_dir> [--mask-dir DIR] [--length-mm X]
Gate:  python tools/sfm_real.py --gate [part]   renders 16 frames to disk and runs the full
       path on them - estimated intrinsics, plane from points, no truth consulted - then
       compares the recovered depth/length ratio against the mesh.
"""
import glob, json, os, shutil, subprocess, sys, tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import carve as C  # noqa: E402
from sfm_check import COLMAP, quat_to_R, read_images, run_colmap  # noqa: E402


def read_camera(model_dir):
    with open(os.path.join(model_dir, "cameras.txt")) as fh:
        f = next(l for l in fh if not l.startswith("#")).split()
    model, w, h = f[1], float(f[2]), float(f[3])
    p = [float(x) for x in f[4:]]
    if model == "SIMPLE_RADIAL":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]])
        dist = np.array([p[3], 0, 0, 0, 0.0])
    elif model == "PINHOLE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        dist = np.zeros(5)
    else:
        raise SystemExit("unhandled camera model %s" % model)
    return K, dist


def read_points(model_dir):
    pts = []
    with open(os.path.join(model_dir, "points3D.txt")) as fh:
        for l in fh:
            if not l.startswith("#") and l.strip():
                f = l.split()
                pts.append([float(f[1]), float(f[2]), float(f[3])])
    return np.array(pts)


def dominant_plane(pts, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    best, best_in = None, 0
    tol = 0.01 * float(np.linalg.norm(np.ptp(pts, axis=0)))
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        d = np.abs((pts - a) @ n)
        inl = int((d < tol).sum())
        if inl > best_in:
            best, best_in = (n, a), inl
    n, a = best
    sel = pts[np.abs((pts - a) @ n) < tol]
    centroid = sel.mean(0)
    u, s, vt = np.linalg.svd(sel - centroid)
    return vt[2], centroid, best_in / len(pts)


def to_desk_frame(images, K, dist, pts):
    n, origin, frac = dominant_plane(pts)
    centres = np.array([-R.T @ t for R, t in images.values()])
    if np.median((centres - origin) @ n) < 0:
        n = -n
    z = n
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])
    height = float(np.median((centres - origin) @ n))
    s = 60.0 / max(height, 1e-9)
    views = []
    for name, (R, t) in sorted(images.items()):
        Rn = R @ Rw.T
        tn = (t + R @ origin) * s
        rvec, _ = cv2.Rodrigues(Rn)
        views.append({"name": name, "rvec": rvec, "tvec": tn.reshape(3, 1), "K": K, "dist": dist})
    return views, frac, s


def hit_points(views):
    hits = []
    for v in views:
        R, _ = cv2.Rodrigues(v["rvec"])
        eye = (-R.T @ v["tvec"]).reshape(3)
        axis = R.T @ np.array([0, 0, 1.0])
        if abs(axis[2]) > 1e-6:
            lam = -eye[2] / axis[2]
            if lam > 0:
                hits.append(eye + lam * axis)
    return np.array(hits)


def measure(image_dir, mask_dir=None, length_mm=None, voxel=0.4):
    work = tempfile.mkdtemp(prefix="sfmreal_")
    model, err = run_colmap(image_dir, work, camera_model="SIMPLE_RADIAL")
    if model is None:
        raise SystemExit("SfM failed: %s (is the surface textured? sixteen views?)" % err)
    images = read_images(model)
    K, dist = read_camera(model)
    pts = read_points(model)
    print("  registered %d images, %d sparse points" % (len(images), len(pts)))
    views, plane_frac, scale = to_desk_frame(images, K, dist, pts)
    print("  desk plane holds %.0f%% of the sparse cloud" % (100 * plane_frac))
    masks = []
    for v in views:
        path = os.path.join(image_dir, v["name"])
        if mask_dir:
            m = cv2.imread(os.path.join(mask_dir, os.path.splitext(v["name"])[0] + ".png"), 0) > 128
        else:
            from photo2fcstd.trace import segment_photo
            m = segment_photo(path)
        masks.append(m)
    hits = hit_points(views)
    centre = np.median(hits, axis=0)
    cams = []
    for v in views:
        R, _ = cv2.Rodrigues(v["rvec"])
        cams.append((-R.T @ v["tvec"]).reshape(3))
    cams = np.array(cams)
    r_xy = float(np.median(np.linalg.norm(cams[:, :2] - centre[:2], axis=1)))
    z_max = 0.9 * float(cams[:, 2].min())
    plane = (float(min(centre[:2]) - r_xy), float(max(centre[:2]) + r_xy))
    rough = C.footprint(views, masks, plane_mm=plane, voxel_mm=r_xy / 60.0, max_height_mm=z_max)
    if rough is None:
        raise SystemExit("no footprint: the masks and the recovered cameras do not agree")
    span = max(rough[0][1] - rough[0][0], rough[1][1] - rough[1][0])
    fine = span / float(os.environ.get('P2F_SFM_RES', 90.0))
    bounds = (rough[0], rough[1], (0.0, z_max))
    carved = C.carve(views, masks, voxel_mm=fine, allow_misses=1, bounds=bounds, max_height_mm=z_max)
    if carved is None:
        raise SystemExit("carve produced nothing: check the masks")
    from photo2fcstd import fuse
    carved["views"] = len(views)
    ext = carved["extents_mm"]
    depth = fuse.measured_depth(carved)
    length = float(max(ext[0], ext[1]))
    print("  carved %d voxels, extents %s units; depth is the median interior top height"
          % (len(carved["points_mm"]), np.round(ext, 2).tolist()))
    print("  depth / length = %.3f" % (depth / length))
    if length_mm:
        mm = length_mm / length
        print("  with length %.1f mm: depth %.2f mm, extents %s mm"
              % (length_mm, depth * mm, np.round(ext * mm, 2).tolist()))
    else:
        print("  scale unknown: give --length-mm from one caliper reading for millimetres")
    return {"ratio": depth / length, "extents": ext.tolist(), "axis": 2,
            "views": len(views), "carved": carved}


def gate(part=None, views=16):
    import trimesh
    import sfm_check as SC
    from photo2fcstd import bench, sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    part = part or next(p for p in sorted(ideal) if SS.trustworthy(ideal[p]))
    mesh = trimesh.load(bench.truth_of(part))
    mesh.apply_translation(-np.array([mesh.bounds[:, 0].mean(), mesh.bounds[:, 1].mean(), mesh.bounds[0][2]]))
    radius = float(np.max(mesh.extents)) * 1.9
    truth = SC.poses(views, radius)
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(30, 226, (600, 600, 3), dtype=np.uint8), (0, 0), 1.4)
    work = tempfile.mkdtemp(prefix="sfmgate_")
    images = os.path.join(work, "images")
    md = os.path.join(work, "masks")
    os.makedirs(images), os.makedirs(md)
    for i, v in enumerate(truth):
        img, m = SC.render(mesh, v, texture, float(np.max(mesh.extents)) * 1.6, rng)
        cv2.imwrite(os.path.join(images, "v%03d.png" % i), img)
        cv2.imwrite(os.path.join(md, "v%03d.png" % i), m.astype(np.uint8) * 255)
    print("gate on %s: %d rendered frames, tool sees only the images\n" % (part, views))
    got = measure(images, mask_dir=md)
    te = mesh.extents
    td, tl = float(min(te)), float(max(te))
    print("\n  truth: extents %s mm, depth/length %.3f -> recovered %.3f (%.1f%% off)"
          % (np.round(te, 2).tolist(), td / tl, got["ratio"], 100 * abs(got["ratio"] - td / tl) / (td / tl)))
    from photo2fcstd import cli, fuse
    doc = fuse.fused_spec(got["carved"], name=part, length_mm=tl)
    sp = os.path.join(work, "fused.spec.json")
    json.dump(doc, open(sp, "w"))
    rep = cli.freecad_build(sp, os.path.join(work, "fused.FCStd"))
    depth_param = doc["outline"]["depth_px"]
    print("  fused FCStd: valid=%s solids=%d, depth in sheet %.2f mm (true %.2f), "
          "sketches unsolved %d, trusted=%s"
          % (rep["valid"], rep["solids"], depth_param, td,
             sum(1 for s in rep["sketches"].values() if s["solve"] != 0),
             doc["outline"]["depth_trusted"]))
    shutil.rmtree(work)


if __name__ == "__main__":
    if "--gate" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--gate"]
        gate(*rest)
    else:
        args = sys.argv[1:]
        length = None
        if "--length-mm" in args:
            i = args.index("--length-mm")
            length = float(args[i + 1]); del args[i:i + 2]
        mask_dir = None
        if "--mask-dir" in args:
            i = args.index("--mask-dir")
            mask_dir = args[i + 1]; del args[i:i + 2]
        measure(args[0], mask_dir=mask_dir, length_mm=length)
