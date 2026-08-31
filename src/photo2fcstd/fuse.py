"""Fuse posed depth maps into a point cloud.

Carving cannot see into a recess: the silhouette of a boxed cavity is the box. Depth
measures the cavity directly, so where depth is available it beats a visual hull on
exactly the geometry the hull is blind to.
"""
import numpy as np


ERODE_PX = 3
DEPTH_MAD = 6.0


def trim(mask, erode_px=ERODE_PX):
    """Pull the mask in from its edge: boundary pixels mix object and background depth."""
    import cv2
    if erode_px <= 0:
        return mask
    k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k) > 0


def backproject(depth_mm, K, mask=None, stride=1, erode_px=ERODE_PX, mad=DEPTH_MAD):
    """Depth map to points in the camera frame, millimetres."""
    h, w = depth_mm.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_mm[::stride, ::stride].astype(float)
    keep = z > 0
    if mask is not None:
        keep &= trim(mask, erode_px)[::stride, ::stride]
    if mad and keep.any():
        zk = z[keep]
        med = np.median(zk)
        spread = np.median(np.abs(zk - med)) or 1.0
        keep &= np.abs(z - med) < mad * spread
    if not keep.any():
        return np.zeros((0, 3))
    x = (xs[keep] - K[0, 2]) * z[keep] / K[0, 0]
    y = (ys[keep] - K[1, 2]) * z[keep] / K[1, 1]
    return np.column_stack([x, y, z[keep]])


def to_object_frame(points_cam, view):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(view["rvec"], float))
    t = np.asarray(view["tvec"], float).reshape(3)
    return (points_cam - t) @ R


def fuse(views, depths, masks, K_key="K", stride=2, voxel_mm=0.5, erode_px=ERODE_PX):
    """Posed depth maps to one deduplicated point cloud in the object frame."""
    chunks = []
    for view, d, m in zip(views, depths, masks):
        if d is None:
            continue
        pts = backproject(d, np.asarray(view[K_key], float), m, stride, erode_px)
        if len(pts):
            chunks.append(to_object_frame(pts, view))
    if not chunks:
        return np.zeros((0, 3))
    pts = np.vstack(chunks)
    keys = np.round(pts / voxel_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def occupancy(points_mm, voxel_mm=0.5):
    return set(map(tuple, np.round(np.asarray(points_mm) / voxel_mm).astype(int)))


def demo():
    K = np.array([[500.0, 0, 100], [0, 500.0, 100], [0, 0, 1]])
    d = np.full((200, 200), 1000.0)
    pts = backproject(d, K)
    assert len(pts) == 200 * 200
    assert abs(pts[:, 2].mean() - 1000.0) < 1e-6
    span = pts[:, 0].max() - pts[:, 0].min()
    assert abs(span - 199 * 1000.0 / 500.0) < 1e-3, span
    print("fuse self-check ok: flat wall at 1000 mm backprojects to %.1f mm across" % span)


if __name__ == "__main__":
    demo()


FREE_SPACE_MM = 2.0


def carve_with_depth(views, masks, depths, voxel_mm=0.8, bounds=None, allow_misses=1,
                     free_space_mm=FREE_SPACE_MM, erode_px=ERODE_PX, min_votes=2):
    """Silhouette carving, then remove voxels the depth maps prove are empty.

    A voxel closer to the camera than the measured surface has nothing in front of it,
    so it is free space. That is how a cavity gets carved: its interior is in front of
    the surface behind it, which no silhouette can express. `min_votes` views must agree
    before a voxel is removed, so one noisy depth pixel cannot delete real surface.
    """
    from photo2fcstd import carve as C
    carved = C.carve(views, masks, voxel_mm=voxel_mm, bounds=bounds, allow_misses=allow_misses)
    if carved is None:
        return None
    pts = carved["points_mm"]
    votes = np.zeros(len(pts), int)
    for view, d, m in zip(views, depths, masks):
        if d is None:
            continue
        uv = C.project(pts, view)
        h, w = d.shape
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        on = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not on.any():
            continue
        # only trust depth well inside the mask: at the silhouette edge the pixel is
        # background, which sits far behind the part and would delete real surface
        inside = trim(m, erode_px) if m is not None else None
        seen = np.zeros(len(pts), float)
        seen[on] = d[v[on], u[on]]
        if inside is not None:
            ok = np.zeros(len(pts), bool)
            ok[on] = inside[v[on], u[on]]
            seen[~ok] = 0.0
        import cv2
        R, _ = cv2.Rodrigues(np.asarray(view["rvec"], float))
        t = np.asarray(view["tvec"], float).reshape(3)
        z = (pts @ R.T + t)[:, 2]
        votes += on & (seen > 0) & (z < seen - free_space_mm)
    alive = votes < min_votes
    kept = pts[alive]
    if not len(kept):
        return carved
    out = dict(carved)
    out["points_mm"] = kept
    out["extents_mm"] = np.ptp(kept, axis=0) + voxel_mm
    out["volume_mm3"] = float(len(kept) * voxel_mm ** 3)
    out["removed_by_depth"] = int((~alive).sum())
    return out
