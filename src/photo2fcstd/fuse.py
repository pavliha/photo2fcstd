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
