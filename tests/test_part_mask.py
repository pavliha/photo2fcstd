import numpy as np
import pytest

from photo2fcstd import capture, capture_check as CK


def test_the_board_quad_follows_the_pose():
    view = CK.board_views(1, radius=max(CK.board_mm()) * 1.6,
                          elevations=CK.DETECTABLE_ELEV[:1])[0]
    img, _ = CK.render_view(view, CK.board_texture())
    got = capture.pose(img)
    quad = capture.board_quad(got)
    assert quad.shape == (4, 2)
    assert np.isfinite(quad).all()


def test_the_board_render_matches_a_part_free_photograph():
    view = CK.board_views(1, radius=max(CK.board_mm()) * 1.6,
                          elevations=CK.DETECTABLE_ELEV[:1])[0]
    img, _ = CK.render_view(view, CK.board_texture())
    got = capture.pose(img)
    expected = capture.board_render(got, img.shape)
    import cv2
    inside = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, capture.board_quad(got).astype(np.int32), 1)
    d = np.abs(img.astype(np.int16) - expected.astype(np.int16)).max(axis=2)
    assert np.median(d[inside > 0]) < 10


def test_part_mask_refuses_rather_than_returning_something_wrong():
    view = CK.board_views(1, radius=max(CK.board_mm()) * 1.6,
                          elevations=CK.DETECTABLE_ELEV[:1])[0]
    img, _ = CK.render_view(view, CK.board_texture())
    with pytest.raises(NotImplementedError):
        capture.part_mask(img, capture.pose(img))
