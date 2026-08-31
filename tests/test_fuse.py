import numpy as np
import pytest

from photo2fcstd import fuse


def test_a_flat_wall_backprojects_to_its_true_width():
    K = np.array([[500.0, 0, 100], [0, 500.0, 100], [0, 0, 1]])
    pts = fuse.backproject(np.full((200, 200), 1000.0), K, mad=0)
    assert len(pts) == 200 * 200
    assert abs(pts[:, 2].mean() - 1000.0) < 1e-6
    assert abs((pts[:, 0].max() - pts[:, 0].min()) - 199 * 2.0) < 1e-3


def test_the_object_frame_round_trips():
    import cv2
    rvec = np.array([[0.3], [-0.2], [0.5]])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[10.0], [-5.0], [600.0]])
    obj = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    cam = obj @ R.T + t.reshape(3)
    back = fuse.to_object_frame(cam, {"rvec": rvec, "tvec": t})
    assert np.abs(back - obj).max() < 1e-9


def test_eroding_pulls_the_mask_in():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    assert fuse.trim(m, 0).sum() == 400
    assert fuse.trim(m, 3).sum() == 14 * 14
