import sys

import cv2
import numpy as np

from photo2fcstd.errors import CaptureError

from photo2fcstd.make_target import DICT, SQUARE_MM, board

PPMM = 20.0


def as_uint8(image):
    """Whatever the caller had, in the 8-bit form the aruco detector requires.

    `trace.load` returns float32 in 0..1 and every capture entry point feeds it straight to the
    detector, which raises. Nothing caught it because the board path had never been run on a real
    photograph - the datasets have none - so the failure only appears at the moment the rig is
    first used.
    """
    a = np.asarray(image)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    top = float(a.max()) if a.size else 0.0
    if top <= 1.001:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def detect(gray):
    gray = as_uint8(gray)
    detector = cv2.aruco.CharucoDetector(board())
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        raise CaptureError("found %d charuco corners, need 6+ - reshoot with the target flat "
                           "and fully in frame" % (0 if ids is None else len(ids)))
    return corners.reshape(-1, 2), ids.reshape(-1)


def board_points(ids):
    per_row, rows = board().getChessboardSize()[0] - 1, board().getChessboardSize()[1]
    return np.array([[((int(i) % per_row) + 1) * SQUARE_MM,
                      (rows - 1 - (int(i) // per_row)) * SQUARE_MM] for i in ids],
                    dtype=np.float32)


def image_points_mm(ids):
    pts = board_points(ids)
    return np.c_[pts[:, 0], board().getChessboardSize()[1] * SQUARE_MM - pts[:, 1]].astype(np.float32)


def rectify(path, out):
    img = cv2.imread(path)
    if img is None:
        raise CaptureError("cannot read %s" % path)
    corners, ids = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    H, _ = cv2.findHomography(corners, image_points_mm(ids) * PPMM, cv2.RANSAC, 3.0)
    h, w = img.shape[:2]
    box = cv2.perspectiveTransform(
        np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32), H)[0]
    lo, hi = box.min(axis=0), box.max(axis=0)
    shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], dtype=np.float64)
    size = np.clip((hi - lo), 1, 6000).astype(int)
    warp = cv2.warpPerspective(img, shift @ H, tuple(size))
    cv2.imwrite(out, warp)
    err = np.linalg.norm(
        cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        - image_points_mm(ids) * PPMM, axis=1).mean() / PPMM
    print("%s -> %s  %d corners, %.2f px/mm, mean reprojection %.3f mm"
          % (path, out, len(ids), PPMM, err))


def demo():
    import tempfile
    from make_target import render
    with tempfile.TemporaryDirectory() as tmp:
        target = "%s/target.png" % tmp
        render(target)
        img = cv2.imread(target)
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[80, 40], [w - 30, 10], [w - 90, h - 60], [40, h - 20]])
        skew = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                   borderValue=(255, 255, 255))
        cv2.imwrite("%s/skew.png" % tmp, skew)
        rectify("%s/skew.png" % tmp, "%s/flat.png" % tmp)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        rectify(sys.argv[1], sys.argv[2])
