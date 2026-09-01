import numpy as np
import pytest

from photo2fcstd.trace import elements, fit_circle, fit_ellipse, merge_collinear, outline, primitives, snap_rectilinear, symmetrize


def disc(n=200, r=60, cx=100, cy=100):
    y, x = np.ogrid[:2 * cx, :2 * cy]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2


def rect(w=120, h=60, pad=20):
    m = np.zeros((h + 2 * pad, w + 2 * pad), bool)
    m[pad:pad + h, pad:pad + w] = True
    return m


def test_circle_fit_recovers_radius():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts = np.c_[50 + 30 * np.cos(t), -20 + 30 * np.sin(t)]
    cx, cy, r, rel, aspect = fit_circle(pts)
    assert abs(r - 30) < 0.5 and abs(cx - 50) < 0.5 and abs(cy + 20) < 0.5 and rel < 0.01


def test_ellipse_fit_reports_aspect():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    f = fit_ellipse(np.c_[100 + 60 * np.cos(t), 100 + 30 * np.sin(t)])
    assert abs(f["aspect"] - 0.5) < 0.05 and f["rms"] < 1.0


def test_outline_finds_hole_and_rectangularity():
    m = rect()
    m[35:45, 60:70] = False
    poly, shape = outline(m)
    assert shape["rectangularity"] > 0.9
    assert len(shape["holes"]) == 1
    assert 0.0 < shape["hole_frac"] < 0.1


def test_primitives_turn_a_disc_into_one_circle():
    poly, shape = outline(disc())
    loops = primitives([shape["raw"]], 200)
    assert len(loops) == 1 and loops[0]["type"] == "circle"
    assert abs(loops[0]["r"] - 60) < 2


def test_primitives_turn_a_rectangle_into_four_snapped_lines():
    poly, shape = outline(rect())
    loops = primitives([shape["raw"]], 160)
    loop = loops[0]
    assert loop["type"] == "loop"
    assert len(loop["elements"]) == 4
    assert sorted(loop["kinds"]) == ["H", "H", "V", "V"]


def test_symmetrize_mirrors_a_nearly_symmetric_mask():
    m = rect()
    m[20:26, 20:24] = False
    out, axes = symmetrize(m)
    assert axes
    for axis, name in ((1, "x"), (0, "y")):
        if name in axes:
            assert (out == np.flip(out, axis=axis)).all()


def test_merge_collinear_drops_redundant_points():
    pts = np.array([[0, 0], [50, 0], [100, 0], [100, 50], [0, 50]], float)
    assert len(merge_collinear(pts, 5.0)) == 4


def test_a_rectangle_is_not_a_circle():
    poly, shape = outline(rect())
    assert len(shape["raw"]) >= 12
    loops = primitives([shape["raw"]], 160)
    assert loops[0]["type"] == "loop"


def test_a_disc_is_still_a_circle():
    poly, shape = outline(disc())
    assert primitives([shape["raw"]], 200)[0]["type"] == "circle"


def test_collinear_lines_merge_even_when_the_loop_has_an_arc():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [10, 0]},
           {"type": "line", "p0": [10, 0], "p1": [20, 0]},
           {"type": "line", "p0": [20, 0], "p1": [30, 0]},
           {"type": "arc", "p0": [30, 0], "p1": [30, 10], "cx": 25, "cy": 5, "r": 7.07, "ccw": True},
           {"type": "line", "p0": [30, 10], "p1": [0, 10]},
           {"type": "line", "p0": [0, 10], "p1": [0, 0]}]
    merged = trace.merge_line_elements(els, min_len=1.0)
    assert sum(1 for e in merged if e["type"] == "line") == 3
    assert sum(1 for e in merged if e["type"] == "arc") == 1
    assert merged[0]["p0"] == [0, 0] and merged[0]["p1"] == [30, 0]


def test_merging_never_swallows_a_real_corner():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [30, 0]},
           {"type": "line", "p0": [30, 0], "p1": [30, 20]},
           {"type": "line", "p0": [30, 20], "p1": [0, 20]},
           {"type": "line", "p0": [0, 20], "p1": [0, 0]}]
    assert len(trace.merge_line_elements(els, min_len=1.0)) == 4


def test_a_tiny_segment_is_absorbed():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [30, 0]},
           {"type": "line", "p0": [30, 0], "p1": [30.4, 3]},
           {"type": "line", "p0": [30.4, 3], "p1": [30, 20]},
           {"type": "line", "p0": [30, 20], "p1": [0, 20]},
           {"type": "line", "p0": [0, 20], "p1": [0, 0]}]
    assert len(trace.merge_line_elements(els, min_len=5.0)) == 4


def line_loop(points):
    return [{"type": "line", "p0": list(points[i]), "p1": list(points[(i + 1) % len(points)])}
            for i in range(len(points))]


def test_a_nearly_rectangular_slot_becomes_an_exact_rectangle():
    from photo2fcstd import trace
    els = line_loop([(0, 0), (40, 1), (41, 20), (39, 19.4), (0.6, 20)])
    out = trace.rectangularise(els)
    assert len(out) == 4
    xs = sorted({round(e["p0"][0], 3) for e in out})
    ys = sorted({round(e["p0"][1], 3) for e in out})
    assert len(xs) == 2 and len(ys) == 2


def test_an_l_shape_is_left_alone():
    from photo2fcstd import trace
    els = line_loop([(0, 0), (40, 0), (40, 10), (15, 10), (15, 30), (0, 30)])
    assert len(trace.rectangularise(els)) == len(els)


def test_a_loop_with_an_arc_is_left_alone():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [10, 0]},
           {"type": "arc", "p0": [10, 0], "p1": [10, 5], "cx": 10, "cy": 2.5, "r": 2.5, "ccw": True},
           {"type": "line", "p0": [10, 5], "p1": [0, 5]},
           {"type": "line", "p0": [0, 5], "p1": [0, 0]}]
    assert trace.rectangularise(els) is els


def test_rectangularise_keeps_the_winding_direction():
    from photo2fcstd import trace
    ccw = line_loop([(0, 0), (40, 0.5), (40, 20), (0, 20)])
    cw = line_loop([(0, 0), (0, 20), (40, 20), (40, 0.5)])
    a, b = trace.rectangularise(ccw), trace.rectangularise(cw)
    assert trace._signed_area(np.array([e["p0"] for e in a], float)) > 0
    assert trace._signed_area(np.array([e["p0"] for e in b], float)) < 0


def test_symmetrize_mirrors_the_holes_not_just_the_outline():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 9:14] = False
    out, axes = symmetrize(mask)
    assert "x" in axes
    assert not out[12:18, 9:14].any()
    assert not out[12:18, 26:31].any()


def test_symmetrize_leaves_an_asymmetric_part_alone():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:20] = True
    mask[8:12, 20:34] = True
    before = mask.sum()
    out, axes = symmetrize(mask)
    assert axes == [] and out.sum() == before


def test_symmetrize_keeps_a_hole_that_is_already_mirrored():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 9:14] = False
    mask[12:18, 26:31] = False
    out, _ = symmetrize(mask)
    assert not out[12:18, 9:14].any() and not out[12:18, 26:31].any()
    assert out[20:30, 15:25].all()


def test_symmetry_reflects_the_half_that_shows_more_of_the_holes():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 8:14] = False
    mask[14:16, 26:30] = False
    out, axes = symmetrize(mask)
    assert "x" in axes
    holes_left = int((~out[12:18, 8:14]).sum())
    holes_right = int((~out[12:18, 26:32]).sum())
    assert holes_left == holes_right and holes_left > 4


def test_reflecting_is_exact_for_odd_and_even_widths():
    from photo2fcstd.trace import symmetrize
    for width in (40, 41):
        mask = np.zeros((40, width), bool)
        mask[5:35, 5:width - 5] = True
        mask[12:18, 9:14] = False
        mask[12:18, width - 14:width - 9] = False
        out, axes = symmetrize(mask)
        assert "x" in axes
        assert (out == np.flip(out, axis=1)).all()
