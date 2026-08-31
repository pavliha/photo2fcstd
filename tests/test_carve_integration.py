import cv2
import numpy as np
import pytest

from photo2fcstd.capture import board_corners, calibrate, pose
from photo2fcstd.carve import carve, project
from photo2fcstd.make_target import render
from photo2fcstd.rectify import board_points


@pytest.fixture(scope="module")
def board_image(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("board") / "target.png")
    render(path)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def skewed(img, dx, dy):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx // 2, dy // 3], [w - dx, h - dy], [dx // 2, h - dy // 2]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(255, 255, 255))


VIEWS = [(60, 30), (110, 45), (150, 70), (200, 90), (40, 20), (90, 60)]
BLOCK = np.array([[x, y, z] for x in (50.0, 70.0) for y in (60.0, 80.0) for z in (0.0, 10.0)])


@pytest.fixture(scope="module")
def posed_silhouettes(board_image):
    views, masks = [], []
    for dx, dy in VIEWS:
        img = skewed(board_image, dx, dy)
        p = pose(img)
        if p is None:
            continue
        uv = project(BLOCK, p).astype(np.float32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.reshape(-1, 1, 2)).astype(np.int32), 1)
        views.append(p)
        masks.append(mask.astype(bool))
    return views, masks


def test_pose_recovers_the_board_it_was_measured_from(board_image):
    for dx, dy in VIEWS:
        img = skewed(board_image, dx, dy)
        p = pose(img)
        corners, ids = board_corners(img)
        assert p is not None and corners is not None
        object_points = np.c_[board_points(ids), np.zeros(len(ids))]
        reprojected = project(object_points, p)
        assert np.linalg.norm(reprojected - corners, axis=1).mean() < 30.0


def test_carve_consumes_capture_poses(posed_silhouettes):
    views, masks = posed_silhouettes
    assert len(views) >= 5
    carved = carve(views, masks, voxel_mm=1.0, allow_misses=0, bounds=((40, 80), (50, 90), (0, 20)))
    assert carved is not None
    footprint = np.sort(carved["extents_mm"][:2])
    assert (footprint > 18.0).all() and (footprint < 25.0).all(), carved["extents_mm"]


def test_height_is_unconstrained_without_grazing_views(posed_silhouettes):
    views, masks = posed_silhouettes
    tall = carve(views, masks, voxel_mm=1.0, allow_misses=0, bounds=((40, 80), (50, 90), (0, 30)))
    short = carve(views, masks, voxel_mm=1.0, allow_misses=0, bounds=((40, 80), (50, 90), (0, 20)))
    assert tall["extents_mm"][2] > short["extents_mm"][2]


def test_calibration_across_views_is_consistent(board_image):
    cal = calibrate([skewed(board_image, dx, dy) for dx, dy in VIEWS])
    assert cal is not None and cal["views"] >= 4
    assert cal["K"][0, 0] > 0 and cal["rms_px"] < 10
