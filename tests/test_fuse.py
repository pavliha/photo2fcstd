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


def test_free_space_needs_more_than_one_view_to_agree():
    """One noisy depth pixel must not delete a voxel; two agreeing views may."""
    import cv2
    K = np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]])
    view = {"rvec": np.zeros((3, 1)), "tvec": np.array([[0.0], [0.0], [100.0]]),
            "K": K, "dist": np.zeros(5)}
    mask = np.ones((100, 100), bool)
    # only the left half reports a distant surface, so only those voxels are free space
    far = np.zeros((100, 100))
    far[:, :50] = 200.0
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    kw = dict(voxel_mm=2.0, bounds=bounds, erode_px=0)
    plain = fuse.carve_with_depth([view], [mask], [np.zeros((100, 100))], min_votes=1, **kw)
    one = fuse.carve_with_depth([view], [mask], [far], min_votes=1, **kw)
    two_needed = fuse.carve_with_depth([view], [mask], [far], min_votes=2, **kw)
    both = fuse.carve_with_depth([view, view], [mask, mask], [far, far], min_votes=2, **kw)
    n = lambda c: len(c["points_mm"])
    assert n(one) < n(plain), "a distant surface should carve the voxels in front of it"
    assert n(two_needed) == n(plain), "one view must not be enough when two votes are required"
    assert n(both) == n(one), "two agreeing views should carve what one view alone could not"
