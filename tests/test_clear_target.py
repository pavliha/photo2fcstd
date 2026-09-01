import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
import cv2

from photo2fcstd import capture, capture_check as CK, make_target


def clear_texture():
    tex = CK.board_texture()
    ph, pw = tex.shape[:2]
    return make_target.clear_centre(tex.copy(), pw, ph)


def scene(size=(26.0, 16.0, 7.0), n=6):
    w, h = CK.board_mm()
    mesh = trimesh.creation.box(size)
    mesh.apply_translation([w / 2, h / 2, size[2] / 2])
    tex = clear_texture()
    out = []
    for v in CK.board_views(n, radius=max(w, h) * 1.6, elevations=CK.DETECTABLE_ELEV):
        img, truth = CK.render_view(v, tex, mesh)
        photo = np.where(img < 250, img, np.full_like(img, 235))
        got = capture.pose(photo)
        if got is not None:
            out.append((photo, got, truth))
    return out


def test_clearing_the_centre_does_not_stop_the_poses():
    shots = scene()
    assert len(shots) == 6


def test_the_clear_patch_sits_inside_the_board():
    v = scene(n=1)[0][1]
    board = capture.board_quad(v)
    clear = capture.clear_quad(v)
    poly = board.astype(np.float32).reshape(-1, 1, 2)
    assert all(cv2.pointPolygonTest(poly, tuple(map(float, p)), False) >= 0 for p in clear)


def test_the_part_is_found_where_a_checkerboard_defeated_every_method():
    shots = scene()
    ious = []
    for photo, view, truth in shots:
        m = capture.part_mask(photo, view)
        ious.append((m & truth).sum() / max((m | truth).sum(), 1))
    assert np.mean(ious) > 0.7, np.mean(ious)


def test_the_threshold_does_not_need_tuning():
    photo, view, truth = scene(n=1)[0]
    got = [((capture.part_mask(photo, view, threshold=t) & truth).sum()
            / max((capture.part_mask(photo, view, threshold=t) | truth).sum(), 1))
           for t in (140, 160, 180)]
    assert max(got) - min(got) < 0.05


def test_an_empty_target_yields_no_part():
    w, h = CK.board_mm()
    v = CK.board_views(1, radius=max(w, h) * 1.6, elevations=CK.DETECTABLE_ELEV[:1])[0]
    img, _ = CK.render_view(v, clear_texture())
    photo = np.where(img < 250, img, np.full_like(img, 235))
    assert capture.part_mask(photo, capture.pose(photo)).sum() == 0
