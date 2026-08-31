import cv2
import numpy as np
import pytest

from photo2fcstd.capture import board_corners, calibrate, pose, rectified
from photo2fcstd.make_target import SQUARE_MM, render
from photo2fcstd.rectify import PPMM


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("t") / "target.png")
    render(path)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def skew(img, dx=90, dy=40):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, dy], [w - dx // 2, dy // 3], [w - dx, h - dy], [dx // 2, h - dy // 2]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(255, 255, 255))


def test_board_is_found_in_a_skewed_photo(target):
    corners, ids = board_corners(skew(target))
    assert corners is not None and len(ids) >= 20


def test_rectify_recovers_metric_scale(target):
    out = rectified(skew(target))
    assert out is not None
    assert out["mm_per_px"] == pytest.approx(1.0 / PPMM)
    assert out["reprojection_mm"] < 0.2


def test_rectified_squares_measure_their_true_size(target):
    out = rectified(skew(target))
    corners, ids = board_corners(out["image"])
    assert corners is not None
    spacing = np.diff(np.unique(np.round(corners[:, 0] / PPMM, 1)))
    assert np.median(spacing[spacing > SQUARE_MM / 2]) == pytest.approx(SQUARE_MM, rel=0.02)


def test_pose_and_calibration(target):
    p = pose(skew(target))
    assert p is not None and p["reprojection_px"] < 30
    cal = calibrate([skew(target, dx, dy) for dx, dy in ((60, 30), (90, 40), (120, 55), (150, 70), (40, 20))])
    assert cal is not None and cal["views"] >= 4 and cal["rms_px"] < 5


def test_missing_board_is_reported_not_guessed():
    blank = np.full((600, 800, 3), 240, np.uint8)
    assert board_corners(blank)[0] is None
    assert rectified(blank) is None
    assert pose(blank) is None
