import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from photo2fcstd import capture_check as CK


def test_the_board_is_laid_out_by_a_rotation_not_a_reflection():
    import cv2
    R, _ = cv2.Rodrigues(CK.board_to_world()[0])
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rendered_board_views_solve_to_the_pose_they_were_rendered_with():
    w, h = CK.board_mm()
    views = CK.board_views(4, radius=max(w, h) * 1.6, elevations=CK.DETECTABLE_ELEV)
    rows = CK.recover(views)
    assert all(r["got"] for r in rows)
    assert max(r["deg"] for r in rows) < 0.5
    assert max(r["mm"] for r in rows) < 1.0


def test_a_grazing_view_is_not_detectable():
    w, h = CK.board_mm()
    rows = CK.recover(CK.board_views(1, radius=max(w, h) * 1.6, elevations=(6.0,)))
    assert rows[0]["got"] is None
