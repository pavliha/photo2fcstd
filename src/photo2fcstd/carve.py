import cv2
import numpy as np

from photo2fcstd.capture import MIN_CALIBRATION_VIEWS, calibrate, pose
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


def top_height(carved, frac=0.9):
    z = carved["points_mm"][:, 2]; vox = carved["voxel_mm"]
    zs = np.arange(z.min(), z.max() + vox, vox)
    area = np.array([((z >= s - vox / 2) & (z < s + vox / 2)).sum() for s in zs])
    ref = np.median(area[:max(3, len(area) // 4)])
    ok = np.nonzero(area >= frac * ref)[0]
    return float(zs[ok[-1]] + vox / 2)


def trimmed_to_top(carved):
    h = top_height(carved)
    pts = carved["points_mm"][carved["points_mm"][:, 2] <= h + 1e-6]
    return {**carved, "points_mm": pts, "top_mm": h, "extents_mm": np.ptp(pts, axis=0) + carved["voxel_mm"],
            "volume_mm3": float(len(pts) * carved["voxel_mm"] ** 3)}


def axis_facing(view):
    R = cv2.Rodrigues(np.asarray(view["rvec"], float))[0]
    return int(np.argmax(np.abs(R[2])))


def view_elevation_deg(view):
    R, _ = cv2.Rodrigues(np.asarray(view["rvec"], float))
    eye = -R.T @ np.asarray(view["tvec"], float).reshape(3)
    r = float(np.linalg.norm(eye))
    return float(np.degrees(np.arcsin(np.clip(eye[2] / max(r, 1e-9), -1.0, 1.0))))


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


PRISM_CONSTANCY = 0.8


def section_constancy(carved, axis):
    """Median slice population over the largest, so 1.0 is a part of constant cross-section.

    A base face only exists if the part is an extrusion of one. The axis classifier was trained
    entirely on PrintCAD, where every part is, and on T-LESS housings it drops to 29% against 33%
    for chance - while getting *more* confident when wrong, so its own score cannot gate it.
    This is measured off the carved volume instead. At 0.8 it leaves PrintCAD untouched, 94% kept
    at the same 89% correct, and on T-LESS it refuses the worst third and lifts the rest from 36%
    to 56% (n=14, small, but the mechanism does not depend on the sample).
    """
    from photo2fcstd import axis_model
    counts = axis_model.slice_counts(carved["points_mm"], carved["voxel_mm"], axis)
    trimmed = counts[1:-1] if len(counts) > 4 else counts
    return float(np.median(trimmed) / max(trimmed.max(), 1)) if len(trimmed) else 0.0


TOP_PPMM = 10.0


def top_face_mask(views, masks, top_mm, bounds_xy, ppmm=TOP_PPMM):
    (x0, x1), (y0, y1) = bounds_xy
    w, h = int(np.ceil((x1 - x0) * ppmm)), int(np.ceil((y1 - y0) * ppmm))
    face = np.ones((h, w), bool)
    for view, mask in zip(views, masks):
        R = cv2.Rodrigues(np.asarray(view["rvec"], float))[0]; t = np.asarray(view["tvec"], float).reshape(3)
        Hp = view["K"] @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * top_mm + t])
        grid = np.array([[1 / ppmm, 0, x0], [0, 1 / ppmm, y0], [0, 0, 1.0]])
        warped = cv2.warpPerspective(mask.astype(np.uint8), Hp @ grid, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0
        face &= warped
    return face


def spec_from_carve(carved, name="part", stl=None, axis=None, views=None, masks=None):
    from photo2fcstd.trace import outline, primitives
    axis = base_axis(carved) if axis is None else axis
    voxel = carved["voxel_mm"]
    if axis == 2 and views is not None and carved.get("top_mm"):
        pts = carved["points_mm"]
        pad = 2.0
        bounds_xy = ((float(pts[:, 0].min()) - pad, float(pts[:, 0].max()) + pad), (float(pts[:, 1].min()) - pad, float(pts[:, 1].max()) + pad))
        from photo2fcstd.trace import upright_mask
        plan, _ = upright_mask(top_face_mask(views, masks, carved["top_mm"], bounds_xy))
        voxel = 1.0 / TOP_PPMM
    else:
        plan = occupancy(carved, axis=axis)
    poly, shape = outline(plan)
    loops_px = [shape["raw"]] + shape["raw_holes"]
    centre = np.array(shape["raw"], float).mean(axis=0)
    centred = [[[(x - centre[0]) * voxel, -(y - centre[1]) * voxel] for x, y in loop] for loop in loops_px]
    length_mm = max(np.ptp(np.array(shape["raw"], float), axis=0)) * voxel
    loops = primitives(centred, length_mm)
    height = float(carved["top_mm"]) if axis == 2 and carved.get("top_mm") else float(np.ptp(carved["points_mm"][:, axis]) + carved["voxel_mm"])
    constancy = section_constancy(carved, axis)
    warning = None
    if constancy < PRISM_CONSTANCY:
        warning = ("this part's cross-section changes along every direction (constancy %.2f), so it "
                   "is not an extrusion of one face and no single sketch describes it - treat the "
                   "chosen view as a guess" % constancy)
    return {"name": name, "mm_per_px": 1.0,
            "scale_note": "metric from the ChArUco board: %d views, %.2f mm voxels" % (carved["views"], voxel),
            "views": {},
            "outline": {"source": ";".join(carved.get("sources", [])) or "carved",
                        "loops": loops, "depth_px": height,
                        "depth_note": "height measured from the carved volume (mm), not guessed",
                        "section_constancy": round(constancy, 3),
                        **({"warning": warning} if warning else {})},
            "revolve": None, "stl": stl}


def revolve_from_carve(carved, name="part", rec=None):
    from photo2fcstd.trace import fit_ellipse, outline
    ext = np.asarray(carved["extents_mm"], float)
    if carved.get("top_mm"):
        ext = ext.copy(); ext[2] = float(carved["top_mm"])
    axis = int(np.argmax(np.abs(ext - np.median(ext))))
    others = [i for i in range(3) if i != axis]
    R = float(np.mean(ext[others]) / 2.0)
    L = float(ext[axis])
    holes = []
    if "bore" in ((rec or {}).get("openings") or []):
        poly, sh = outline(occupancy(carved, axis=axis))
        inner = [fit_ellipse(np.asarray(h, float)) for h in sh["holes"] if len(h) >= 8]
        inner = [f for f in inner if f and 0.05 < f["a"] * carved["voxel_mm"] / R < 0.95]
        if inner:
            r = max(f["a"] for f in inner) * carved["voxel_mm"]
            holes.append({"type": "circle", "cx": 0.0, "cy": 0.0, "r": round(float(r), 2), "source": "board hull, seen through the bore"})
    return {"name": name, "mode": "revolve", "mm_per_px": 1.0, "unit": "mm",
            "scale_note": "metric from the ChArUco board: %d views, %.2f mm voxels" % (carved["views"], carved["voxel_mm"]),
            "views": {}, "outline": None, "stl": None, "measured": [],
            "revolve": {"generic": True, "profile": [[0.0, 0.0], [round(R, 2), 0.0], [round(R, 2), round(L, 2)], [0.0, round(L, 2)]],
                        "holes": holes, "rings": [], "axis": axis,
                        "source": "radius and length from the board hull (measured)"}}


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


def board_views(paths):
    from photo2fcstd.capture import board_mask, camera_matrix, focal_px_from_exif
    from photo2fcstd.trace import load
    images = [load(p) for p in paths]
    cal = calibrate(images) if len(images) >= MIN_CALIBRATION_VIEWS else None
    views, masks, used = [], [], []
    for path, image in zip(paths, images):
        K = cal["K"] if cal else camera_matrix(image.shape, focal_px_from_exif(path, image.shape))
        p = pose(image, K, cal["dist"] if cal else None)
        if p is None:
            continue
        m = board_mask(image, p)
        if m.sum() < 50:
            continue
        views.append(p)
        masks.append(m)
        used.append(path)
    return views, masks, used, cal


def from_known_poses(paths, views, voxel_mm=VOXEL_MM):
    from photo2fcstd.capture import board_mask
    from photo2fcstd.trace import load
    masks = [board_mask(load(p), v) for p, v in zip(paths, views)]
    keep = [i for i, m in enumerate(masks) if m.sum() >= 50]
    views, masks, used = [views[i] for i in keep], [masks[i] for i in keep], [paths[i] for i in keep]
    if len(views) < MIN_POSED_VIEWS:
        raise CaptureError("need at least %d posed views with a visible part, found %d" % (MIN_POSED_VIEWS, len(views)))
    carved = trimmed_to_top(carve(views, masks, voxel_mm, allow_misses=0))
    carved["board_views"], carved["board_masks"] = views, masks
    carved["min_elevation_deg"] = min(view_elevation_deg(v) for v in views)
    carved["sources"] = used
    carved["calibration_rms_px"] = None
    return carved


def from_photos(paths, segment_fn=None, voxel_mm=VOXEL_MM):
    views, masks, used, cal = board_views(paths)
    if len(views) < MIN_POSED_VIEWS:
        raise CaptureError("need the ChArUco target visible in at least %d photos, found %d"
                           % (MIN_POSED_VIEWS, len(views)))
    carved = trimmed_to_top(carve(views, masks, voxel_mm, allow_misses=0))
    carved["board_views"], carved["board_masks"] = views, masks
    carved["min_elevation_deg"] = min(view_elevation_deg(v) for v in views)
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
