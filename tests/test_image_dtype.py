import numpy as np
import pytest

from photo2fcstd.rectify import as_uint8


def test_float_zero_to_one_is_scaled():
    a = as_uint8(np.full((4, 4, 3), 0.5, np.float32))
    assert a.dtype == np.uint8 and 126 <= a[0, 0, 0] <= 129


def test_float_zero_to_255_is_kept():
    a = as_uint8(np.full((4, 4, 3), 200.0, np.float32))
    assert a.dtype == np.uint8 and a[0, 0, 0] == 200


def test_uint8_passes_through_untouched():
    src = np.full((4, 4, 3), 77, np.uint8)
    assert as_uint8(src) is src


def test_the_capture_path_accepts_what_trace_load_returns(tmp_path):
    from photo2fcstd import capture, capture_check as CK
    import cv2
    from photo2fcstd.trace import load
    p = tmp_path / "board.png"
    img, _ = CK.render_view(CK.board_views(1, radius=max(CK.board_mm()) * 1.6,
                                           elevations=CK.DETECTABLE_ELEV[:1])[0],
                            CK.board_texture())
    cv2.imwrite(str(p), img)
    got = capture.pose(load(str(p)))
    assert got is not None, "the board path must accept a float image from trace.load"
    assert got["corners"] > 6
