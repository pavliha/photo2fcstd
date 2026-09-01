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
    assert axes and out.sum() >= m.sum()


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
