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
