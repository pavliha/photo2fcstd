import numpy as np
import pytest

from photo2fcstd import capture, capture_check as CK, make_target


def cleared_texture():
    tex = CK.board_texture()
    ph, pw = tex.shape[:2]
    return make_target.clear_centre(tex.copy(), pw, ph)


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


def test_a_bare_target_yields_no_part():
    """part_mask used to raise, because no appearance method could separate a part from the
    checkerboard. Clearing the target's middle made the problem go away; an empty target should
    still yield nothing."""
    view = CK.board_views(1, radius=max(CK.board_mm()) * 1.6,
                          elevations=CK.DETECTABLE_ELEV[:1])[0]
    img, _ = CK.render_view(view, cleared_texture())
    assert capture.part_mask(img, capture.pose(img)).sum() == 0
