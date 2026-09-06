import os

import cv2
import numpy as np
import pytest

from photo2fcstd import capture, carve, make_target


def _camera(w, h, f):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def _look_at(eye, target=(52.5, 75.0, 0.0)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ eye


def _render(box_mm, R, t, K, size):
    art = make_target.board().generateImage((int(make_target.COLS * make_target.SQUARE_MM * 8), int(make_target.ROWS * make_target.SQUARE_MM * 8)))
    art = make_target.clear_centre(art, art.shape[1], art.shape[0])
    W = make_target.COLS * make_target.SQUARE_MM; Hh = make_target.ROWS * make_target.SQUARE_MM
    src = np.float32([[0, 0], [art.shape[1], 0], [art.shape[1], art.shape[0]], [0, art.shape[0]]])
    corners = np.float32([[0, Hh, 0], [W, Hh, 0], [W, 0, 0], [0, 0, 0]])
    uv = (K @ (corners @ R.T + t).T).T; uv = uv[:, :2] / uv[:, 2:3]
    img = cv2.warpPerspective(cv2.cvtColor(art, cv2.COLOR_GRAY2RGB), cv2.getPerspectiveTransform(src, uv.astype(np.float32)), size, borderValue=(235, 235, 235))
    cx, cy = W / 2, Hh / 2
    bx, by, bz = box_mm
    V = np.array([[cx + sx * bx / 2, cy + sy * by / 2, z] for z in (0.0, bz) for sx in (-1, 1) for sy in (-1, 1)])
    p = (K @ (V @ R.T + t).T).T; p = (p[:, :2] / p[:, 2:3]).astype(np.int32)
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    for f in faces:
        cv2.fillConvexPoly(img, p[f], (40, 90, 200))
    return img


@pytest.mark.parametrize("box", [(30.0, 18.0, 9.0)])
def test_box_on_board_is_measured_from_rendered_photos(box, tmp_path):
    size = (1600, 1200); K = _camera(size[0], size[1], 1500.0)
    centre = np.array([52.5, 75.0, 0.0])
    views, masks = [], []
    for k, (az, el) in enumerate([(0, 30), (90, 30), (180, 25), (270, 35), (45, 55), (225, 15)]):
        d = 260.0
        eye = centre + d * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
        R, t = _look_at(eye)
        img = _render(box, R, t, K, size)
        path = str(tmp_path / ("v%d.jpg" % k)); cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        view = capture.pose(img, K)
        assert view is not None and view["reprojection_px"] < 1.5, view
        views.append(view); masks.append(capture.board_mask(img, view))
    carved = carve.trimmed_to_top(carve.carve(views, masks, voxel_mm=0.5, allow_misses=0))
    got = np.sort(carved["extents_mm"])[::-1]; want = np.sort(box)[::-1]
    assert np.all(np.abs(got - want) < 2.0), (got, want)
    assert abs(carved["top_mm"] - box[2]) < 0.5, carved["top_mm"]
    spec = carve.spec_from_carve(carved, "box")
    assert abs(spec["outline"]["depth_px"] - box[2]) < 0.5, spec["outline"]["depth_px"]
    assert spec["outline"]["loops"][0]["type"] == "loop" and len(spec["outline"]["loops"][0]["elements"]) == 4, spec["outline"]["loops"][0]


FREECAD = os.path.expanduser(os.environ.get("FREECADCMD", "~/Code/FreeCAD/build/release/bin/FreeCADCmd"))


@pytest.mark.skipif(not os.path.exists(FREECAD), reason="FreeCADCmd not present")
def test_design_takes_the_board_path_and_builds_metric(tmp_path):
    from photo2fcstd import design
    box = (30.0, 18.0, 9.0); size = (1600, 1200); K = _camera(size[0], size[1], 1500.0); centre = np.array([52.5, 75.0, 0.0])
    photos = []
    for k, (az, el) in enumerate([(0, 30), (90, 30), (180, 25), (270, 35), (45, 55), (225, 15)]):
        eye = centre + 260.0 * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
        R, t = _look_at(eye)
        path = str(tmp_path / ("v%d.jpg" % k)); cv2.imwrite(path, cv2.cvtColor(_render(box, R, t, K, size), cv2.COLOR_RGB2BGR)); photos.append(path)
    res = design.design(photos, str(tmp_path / "box.FCStd"), rec={"single_extrusion": True, "part_class": None})
    assert res.get("model", 1) is not None, res
    assert res["tier"] == "board" and res["board"]["views"] == 6, res
    assert abs(res["board"]["top_mm"] - 9.0) < 0.5, res["board"]
    assert res["verify"]["silhouette_iou"] >= 0.8, res["verify"]
