import cv2
import numpy as np
import pytest

from photo2fcstd.capture import board_corners
from photo2fcstd.make_target import COLS, DPI, MM_PER_INCH, ROWS, SQUARE_MM, board_image, pdf, render, verify


def cell_mm(art, dpi=DPI):
    corners, ids = board_corners(cv2.cvtColor(art, cv2.COLOR_GRAY2RGB))
    assert corners is not None
    xs = np.unique(np.round(corners[:, 0] / 2.0) * 2.0)
    return float(np.median(np.diff(xs))) / (dpi / MM_PER_INCH)


def test_board_image_draws_squares_of_the_stated_size():
    px = int(round(SQUARE_MM * DPI / MM_PER_INCH))
    assert cell_mm(board_image(px)) == pytest.approx(SQUARE_MM, abs=0.2)


def test_the_pdf_prints_at_the_stated_size(tmp_path):
    path = str(tmp_path / "target.pdf")
    pdf(path)
    measured = verify(path)
    assert measured is not None
    assert measured == pytest.approx(SQUARE_MM, abs=0.3)


def test_the_board_carries_every_inner_corner(tmp_path):
    px = int(round(SQUARE_MM * DPI / MM_PER_INCH))
    corners, ids = board_corners(cv2.cvtColor(board_image(px), cv2.COLOR_GRAY2RGB))
    assert len(ids) == (COLS - 1) * (ROWS - 1)


def test_square_size_can_be_set_for_a_screen_or_an_odd_print(monkeypatch):
    import importlib
    import photo2fcstd.make_target as mt
    monkeypatch.setenv("P2F_SQUARE_MM", "12.7")
    importlib.reload(mt)
    try:
        assert mt.SQUARE_MM == pytest.approx(12.7)
        assert mt.MARKER_MM == pytest.approx(12.7 * 11.0 / 15.0)
    finally:
        monkeypatch.delenv("P2F_SQUARE_MM")
        importlib.reload(mt)
    assert mt.SQUARE_MM == pytest.approx(mt.NOMINAL_SQUARE_MM)


def test_the_screen_target_is_detectable(tmp_path):
    import cv2
    from photo2fcstd.make_target import screen
    path, cell = screen(str(tmp_path / "screen.png"))
    img = cv2.imread(path)
    corners, ids = board_corners(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert corners is not None and len(ids) == (COLS - 1) * (ROWS - 1)
