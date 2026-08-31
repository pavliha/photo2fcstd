import json
import os

import cv2
import numpy as np
import pytest
import trimesh

from photo2fcstd import bop


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("bop")
    obj = 3
    size = (24.0, 16.0, 10.0)
    mesh = trimesh.creation.box(size)
    models = root / "models_cad"
    models.mkdir()
    mesh.export(str(models / ("obj_%06d.ply" % obj)))
    scene_dir = root / "train_primesense" / ("%06d" % obj)
    (scene_dir / "rgb").mkdir(parents=True)
    K = np.array([[1075.0, 0, 640.0], [0, 1075.0, 512.0], [0, 0, 1.0]])
    cameras, poses = {}, {}
    for i in range(24):
        a = np.radians(i * 15.0)
        tilt = np.radians(35 + 30 * (i % 3))
        eye = np.array([260 * np.cos(a) * np.sin(tilt), 260 * np.sin(a) * np.sin(tilt), 260 * np.cos(tilt)])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, [0, 0, 1.0])
        right = right / np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])
        t = (-R @ eye).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        uv, _ = cv2.projectPoints(mesh.vertices, rvec, t, K, np.zeros(5))
        img = np.zeros((1024, 1280, 3), np.uint8)
        cv2.fillConvexPoly(img, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), (200, 200, 200))
        cv2.imwrite(str(scene_dir / "rgb" / ("%06d.png" % i)), img)
        cameras[str(i)] = {"cam_K": K.ravel().tolist(), "depth_scale": 0.1}
        poses[str(i)] = [{"cam_R_m2c": R.ravel().tolist(), "cam_t_m2c": t.ravel().tolist(), "obj_id": obj}]
    json.dump(cameras, open(scene_dir / "scene_camera.json", "w"))
    json.dump(poses, open(scene_dir / "scene_gt.json", "w"))
    return str(root), obj, mesh


def test_views_are_parsed_with_their_poses(scene, monkeypatch):
    root, obj, mesh = scene
    monkeypatch.setattr(bop, "BOP_ROOT", root)
    views = bop.views_of(bop.scene_dir("tless", "train_primesense", obj))
    assert len(views) == 24
    for v in views:
        assert v["K"].shape == (3, 3) and os.path.exists(v["image"])
        assert np.linalg.norm(v["tvec"]) == pytest.approx(260.0, rel=0.02)


def test_the_object_is_segmented_from_the_dark_background(scene, monkeypatch):
    root, obj, mesh = scene
    monkeypatch.setattr(bop, "BOP_ROOT", root)
    path = bop.scene_dir("tless", "train_primesense", obj)
    mask = bop.object_mask(bop.views_of(path)[0]["image"])
    assert mask.any() and mask.mean() < 0.4


def test_carving_a_bop_object_recovers_its_millimetres(scene, monkeypatch):
    root, obj, mesh = scene
    monkeypatch.setattr(bop, "BOP_ROOT", root)
    carved = bop.carve_object(obj, views=24, voxel_mm=1.0, bound_mm=40.0, allow_misses=0)
    row = bop.compare_to_truth(carved, obj)
    assert row["truth_mm"] == [pytest.approx(x, abs=0.01) for x in sorted(mesh.extents)]
    assert row["worst_mm"] < 4.0
    assert 0.9 < row["volume_ratio"] < 2.0


def test_a_missing_object_says_so(scene, monkeypatch):
    from photo2fcstd.errors import CaptureError
    monkeypatch.setattr(bop, "BOP_ROOT", scene[0])
    with pytest.raises(CaptureError, match="no scene for object"):
        bop.carve_object(99)
