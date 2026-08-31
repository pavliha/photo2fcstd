import cv2
import numpy as np

from photo2fcstd.capture import calibrate, pose
from photo2fcstd.errors import CaptureError
from photo2fcstd.make_target import COLS, ROWS, SQUARE_MM

VOXEL_MM = 0.4
MAX_HEIGHT_MM = 120.0
MARGIN_MM = 4.0
MIN_POSED_VIEWS = 3
DEPTH_TOLERANCE_MM = 6.0
DEPTH_VOTES = 0.25


def board_extent_mm():
    return max(COLS, ROWS) * SQUARE_MM


def grid(bounds_mm, voxel_mm):
    (x0, x1), (y0, y1), (z0, z1) = bounds_mm
    xs = np.arange(x0, x1, voxel_mm)
    ys = np.arange(y0, y1, voxel_mm)
    zs = np.arange(z0, z1, voxel_mm)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1), (xs, ys, zs)


def project(points, view):
    projected, _ = cv2.projectPoints(points.reshape(-1, 1, 3).astype(np.float64),
                                     view["rvec"], view["tvec"], view["K"], view["dist"])
    return projected.reshape(-1, 2)


def inside(mask, uv):
    h, w = mask.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hit = np.zeros(len(uv), bool)
    hit[ok] = mask[v[ok], u[ok]]
    return hit, ok


def footprint(views, masks, plane_mm=None, voxel_mm=2.0, max_height_mm=MAX_HEIGHT_MM):
    lo, hi = plane_mm or (0.0, board_extent_mm())
    xs = np.arange(lo, hi, voxel_mm)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)
    votes = np.zeros(len(pts), int)
    for view, mask in zip(views, masks):
        hit, ok = inside(mask, project(pts, view))
        votes += hit
    keep = pts[votes >= max(1, len(views) - 1)]
    if not len(keep):
        return None
    return ((keep[:, 0].min() - MARGIN_MM, keep[:, 0].max() + MARGIN_MM),
            (keep[:, 1].min() - MARGIN_MM, keep[:, 1].max() + MARGIN_MM),
            (0.0, max_height_mm))


def camera_depth(points, view):
    R = cv2.Rodrigues(np.asarray(view["rvec"], float))[0]
    t = np.asarray(view["tvec"], float).reshape(3)
    return (np.asarray(points, float) @ R.T + t)[:, 2]


def in_front_of_surface(points, view, depth_map, depth_scale, tolerance_mm):
    uv = project(points, view)
    h, w = depth_map.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    measured = np.zeros(len(points))
    measured[ok] = depth_map[v[ok], u[ok]] * depth_scale
    seen = ok & (measured > 0)
    return seen & (camera_depth(points, view) < measured - tolerance_mm), seen


def carve(views, masks, voxel_mm=VOXEL_MM, allow_misses=1, bounds=None, max_height_mm=MAX_HEIGHT_MM,
          depths=None, depth_scale=1.0, depth_tolerance_mm=DEPTH_TOLERANCE_MM, depth_votes=DEPTH_VOTES):
    bounds = bounds or footprint(views, masks, max_height_mm=max_height_mm)
    if bounds is None:
        return None
    points, axes = grid(bounds, voxel_mm)
    flat = points.reshape(-1, 3)
    votes = np.zeros(len(flat), int)
    seen = np.zeros(len(flat), int)
    for view, mask in zip(views, masks):
        hit, ok = inside(mask, project(flat, view))
        votes += hit
        seen += ok
    keep = (votes >= np.maximum(seen - allow_misses, 1)) & (seen > 0)
    if depths is not None:
        empty = np.zeros(len(flat), int)
        for view, depth_map in zip(views, depths):
            if depth_map is None:
                continue
            ahead, _ = in_front_of_surface(flat, view, depth_map, depth_scale, depth_tolerance_mm)
            empty += ahead
        keep &= empty < np.maximum(depth_votes * np.maximum(seen, 1), allow_misses + 1)
    occupied = flat[keep]
    if not len(occupied):
        return None
    return {"points_mm": occupied, "voxel_mm": voxel_mm, "axes": axes, "used_depth": depths is not None,
            "extents_mm": np.ptp(occupied, axis=0) + voxel_mm, "views": len(views),
            "volume_mm3": float(len(occupied) * voxel_mm ** 3)}


def occupancy(carved, axis=2):
    pts = carved["points_mm"]
    voxel = carved["voxel_mm"]
    keep = [i for i in range(3) if i != axis]
    idx = np.round((pts[:, keep] - pts[:, keep].min(axis=0)) / voxel).astype(int)
    grid_2d = np.zeros(idx.max(axis=0) + 3, bool)
    grid_2d[idx[:, 0] + 1, idx[:, 1] + 1] = True
    return grid_2d


def base_axis(carved):
    """Look down the axis whose projection shows the part's face.

    A plate is extruded along its short axis but a rod along its long one, so no rule
    based on extent can be right for both: four of them all stall near 70% agreement
    with the best of the three projections. A classifier scoring each axis separately
    reaches 86% and 0.713 sketch IoU against the thinnest-extent rule's 0.634, on an
    oracle ceiling of 0.749. Falls back to thinnest extent when no model is installed.
    """
    from photo2fcstd import axis_model
    learned = axis_model.predict_axis(carved)
    if learned is not None:
        return learned
    pts = carved["points_mm"]
    return int(np.argmin([np.ptp(pts[:, a]) for a in (0, 1, 2)]))


def spec_from_carve(carved, name="part", stl=None, axis=None):
    from photo2fcstd.trace import outline, primitives
    axis = base_axis(carved) if axis is None else axis
    plan = occupancy(carved, axis=axis)
    poly, shape = outline(plan)
    loops_px = [shape["raw"]] + shape["raw_holes"]
    centre = np.array(shape["raw"], float).mean(axis=0)
    voxel = carved["voxel_mm"]
    centred = [[[(x - centre[0]) * voxel, -(y - centre[1]) * voxel] for x, y in loop] for loop in loops_px]
    length_mm = max(np.ptp(np.array(shape["raw"], float), axis=0)) * voxel
    loops = primitives(centred, length_mm)
    height = float(np.ptp(carved["points_mm"][:, axis]) + voxel)
    return {"name": name, "mm_per_px": 1.0,
            "scale_note": "metric from the ChArUco board: %d views, %.2f mm voxels" % (carved["views"], voxel),
            "views": {},
            "outline": {"source": ";".join(carved.get("sources", [])) or "carved",
                        "loops": loops, "depth_px": height,
                        "depth_note": "height measured from the carved volume (mm), not guessed"},
            "revolve": None, "stl": stl}


def mesh_of(carved):
    import trimesh
    from skimage import measure
    voxel = carved["voxel_mm"]
    pts = carved["points_mm"]
    idx = np.round((pts - pts.min(axis=0)) / voxel).astype(int)
    volume = np.zeros(idx.max(axis=0) + 3, bool)
    volume[idx[:, 0] + 1, idx[:, 1] + 1, idx[:, 2] + 1] = True
    verts, faces, _, _ = measure.marching_cubes(volume.astype(float), 0.5)
    mesh = trimesh.Trimesh((verts - 1) * voxel + pts.min(axis=0), faces, process=True)
    mesh.fix_normals()
    return mesh


def from_photos(paths, segment_fn, voxel_mm=VOXEL_MM):
    from photo2fcstd.trace import load
    images = [load(p) for p in paths]
    cal = calibrate(images)
    K = cal["K"] if cal else None
    dist = cal["dist"] if cal else None
    views, masks, used = [], [], []
    for path, image in zip(paths, images):
        p = pose(image, K, dist)
        if p is None:
            continue
        views.append(p)
        masks.append(segment_fn(path))
        used.append(path)
    if len(views) < MIN_POSED_VIEWS:
        raise CaptureError("need the ChArUco target visible in at least %d photos, found %d"
                           % (MIN_POSED_VIEWS, len(views)))
    carved = carve(views, masks, voxel_mm)
    carved["sources"] = used
    carved["calibration_rms_px"] = cal["rms_px"] if cal else None
    return carved


def demo():
    import trimesh
    truth = trimesh.creation.box((20.0, 12.0, 6.0))
    truth.apply_translation([40.0, 40.0, 3.0])
    K = np.array([[1800.0, 0, 640.0], [0, 1800.0, 480.0], [0, 0, 1.0]])
    views, masks = [], []
    for angle, tilt in ((0, 55), (90, 50), (180, 60), (270, 45), (45, 35), (225, 40), (30, 82), (150, 79), (300, 84)):
        a, t = np.radians(angle), np.radians(tilt)
        eye = np.array([40 + 160 * np.cos(a) * np.sin(t), 40 + 160 * np.sin(a) * np.sin(t), 160 * np.cos(t)])
        forward = (np.array([40.0, 40.0, 3.0]) - eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        rvec, _ = cv2.Rodrigues(R)
        tvec = (-R @ eye).reshape(3, 1)
        view = {"rvec": rvec, "tvec": tvec, "K": K, "dist": np.zeros(5)}
        uv = project(truth.vertices, view)
        mask = np.zeros((960, 1280), bool)
        hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask_u8 := np.zeros((960, 1280), np.uint8), hull.astype(np.int32), 1)
        mask |= mask_u8.astype(bool)
        views.append(view)
        masks.append(mask)
    carved = carve(views, masks, voxel_mm=0.5, allow_misses=0, bounds=((25, 60), (25, 60), (0, 20)))
    got = np.sort(carved["extents_mm"])
    want = np.sort(np.array([20.0, 12.0, 6.0]))
    assert np.allclose(got, want, atol=1.5), (got, want)
    print("carve self-check ok: %d voxels, extents %s mm (truth %s)"
          % (len(carved["points_mm"]), np.round(got, 1), want))


if __name__ == "__main__":
    demo()
