import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from photo2fcstd import carve as C
from photo2fcstd import carve_check as CC


def test_camera_poses_are_rotations():
    for v in CC.poses(8, 60.0):
        import cv2
        R, _ = cv2.Rodrigues(v["rvec"])
        assert abs(np.linalg.det(R) - 1) < 1e-6


def test_a_box_carves_to_its_own_size():
    box = trimesh.creation.box((20.0, 12.0, 4.0))
    box.apply_translation(-np.array([0.0, 0.0, box.bounds[0][2]]))
    views = CC.poses(14, float(np.max(box.extents)) * 3.2, (6.0, 18.0, 35.0, 60.0))
    masks = [CC.silhouette(box, v) for v in views]
    bounds = [(float(box.bounds[0][i]) - 1, float(box.bounds[1][i]) + 1) for i in range(2)]
    bounds += [(0.0, float(box.bounds[1][2]) + 1)]
    carved = C.carve(views, masks, voxel_mm=0.25, bounds=bounds)
    assert carved is not None
    got = np.sort(carved["extents_mm"])[::-1]
    want = np.sort(box.extents)[::-1]
    assert np.abs(got - want).max() < 1.5, (got, want)
