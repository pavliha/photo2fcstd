import numpy as np

from photo2fcstd import preflight as P


def posed(elev, azim, err=0.5, mask=0.2):
    return [{"path": "p%d" % i, "board": True, "mask": mask, "elevation": e, "azimuth": a,
             "reprojection_px": err, "corners": 40} for i, (e, a) in enumerate(zip(elev, azim))]


def test_evenly_spaced_cameras_leave_a_small_gap():
    assert P.largest_gap([0, 90, 180, 270]) == 90.0
    assert P.largest_gap([0, 20, 40, 60]) == 300.0


def test_a_single_camera_is_a_full_gap():
    assert P.largest_gap([10]) == 360.0


def test_cameras_bunched_on_one_side_are_rejected():
    rows = posed([30] * 8, np.linspace(0, 60, 8))
    problems, _ = P.verdict(rows)
    assert any("never look from" in p for p in problems)


def test_too_few_posed_views_is_the_first_complaint():
    rows = posed([30] * 3, [0, 120, 240])
    problems, _ = P.verdict(rows)
    assert problems and "solve a pose" in problems[0]


def test_a_high_only_capture_is_called_out_with_the_height_cost():
    rows = posed([55] * 12, np.linspace(0, 330, 12))
    problems, _ = P.verdict(rows, part_width_mm=20.0)
    assert any("Height is bounded by the lowest view" in p for p in problems)
    assert any("mm" in p for p in problems)


def test_a_good_capture_passes():
    rows = posed(list(np.linspace(16, 55, 16)), np.linspace(0, 337, 16))
    problems, _ = P.verdict(rows, part_width_mm=20.0)
    assert problems == []


def test_a_wobbly_board_is_caught():
    rows = posed(list(np.linspace(16, 55, 16)), np.linspace(0, 337, 16), err=6.0)
    problems, _ = P.verdict(rows)
    assert any("reprojects" in p for p in problems)


def test_a_part_that_barely_appears_is_caught():
    rows = posed(list(np.linspace(16, 55, 16)), np.linspace(0, 337, 16), mask=0.0001)
    problems, _ = P.verdict(rows)
    assert any("tiny or missing" in p for p in problems)
