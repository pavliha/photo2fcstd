import cv2
import numpy as np

from photo2fcstd.make_target import SQUARE_MM, board
from photo2fcstd.rectify import PPMM, board_points, detect

MIN_CORNERS = 6
PHONE_FOCAL_FRACTION = 1.2
MIN_CALIBRATION_VIEWS = 4


def board_corners(image):
    from photo2fcstd.rectify import as_uint8
    arr = as_uint8(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    detector = cv2.aruco.CharucoDetector(board())
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None, None
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1)


def homography(corners, ids):
    H, _ = cv2.findHomography(corners, board_points(ids) * PPMM, cv2.RANSAC, 3.0)
    return H


def reprojection_mm(H, corners, ids):
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.linalg.norm(projected - board_points(ids) * PPMM, axis=1).mean() / PPMM)


def rectified(image, pad_mm=5.0):
    arr = np.asarray(image)
    corners, ids = board_corners(arr)
    if corners is None:
        return None
    H = homography(corners, ids)
    h, w = arr.shape[:2]
    box = cv2.perspectiveTransform(np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32), H)[0]
    lo, hi = box.min(axis=0) - pad_mm * PPMM, box.max(axis=0) + pad_mm * PPMM
    shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
    size = np.clip(hi - lo, 1, 8000).astype(int)
    warped = cv2.warpPerspective(arr, shift @ H, tuple(size))
    return {"image": warped, "mm_per_px": 1.0 / PPMM, "corners": len(ids),
            "reprojection_mm": reprojection_mm(H, corners, ids)}


def camera_matrix(shape, focal_px=None):
    h, w = shape[:2]
    f = focal_px or PHONE_FOCAL_FRACTION * max(h, w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], float)


def pose(image, K=None, dist=None):
    from photo2fcstd.rectify import as_uint8
    arr = as_uint8(image)
    corners, ids = board_corners(arr)
    if corners is None:
        return None
    K = camera_matrix(arr.shape) if K is None else K
    dist = np.zeros(5) if dist is None else dist
    object_points = np.c_[board_points(ids), np.zeros(len(ids))].astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1).mean())
    return {"rvec": rvec, "tvec": tvec, "K": K, "dist": dist, "corners": len(ids), "reprojection_px": error}


def calibrate(images):
    detector = cv2.aruco.CharucoDetector(board())
    object_points, image_points, size = [], [], None
    for image in images:
        arr = np.asarray(image)
        size = (arr.shape[1], arr.shape[0])
        corners, ids = board_corners(arr)
        if corners is None or len(ids) < 8:
            continue
        object_points.append(np.c_[board_points(ids), np.zeros(len(ids))].astype(np.float32))
        image_points.append(corners.reshape(-1, 1, 2))
    if len(object_points) < MIN_CALIBRATION_VIEWS:
        return None
    ok, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    return {"K": K, "dist": dist, "rms_px": float(ok), "views": len(object_points)}


def demo():
    from photo2fcstd.make_target import render
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "target.png")
        render(path)
        flat = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = flat.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[90, 40], [w - 40, 15], [w - 100, h - 70], [50, h - 25]])
        skewed = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(255, 255, 255))
        out = rectified(skewed)
        assert out and out["reprojection_mm"] < 0.2, out
        p = pose(skewed)
        assert p and p["reprojection_px"] < 25, p
        print("capture self-check ok: %d corners, %.3f mm reprojection, %.2f px pose error"
              % (out["corners"], out["reprojection_mm"], p["reprojection_px"]))


if __name__ == "__main__":
    demo()
