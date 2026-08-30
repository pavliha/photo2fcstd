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
