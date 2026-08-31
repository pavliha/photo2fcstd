import numpy as np
from shapely.geometry import Polygon

from photo2fcstd.sketch_score import difficulty, equivalent_ellipse, skill, trivial_score


def ring(record_pts):
    return {"loops": [[{"xy": record_pts}]], "n_loops": 1}


def circle_pts(n=64, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)]).tolist()


def test_a_circle_is_free():
    assert trivial_score(ring(circle_pts())) > 0.97
    assert difficulty(ring(circle_pts())) < 0.05


def test_a_cross_is_not_free():
    c = [(-3, -1), (-1, -1), (-1, -3), (1, -3), (1, -1), (3, -1),
         (3, 1), (1, 1), (1, 3), (-1, 3), (-1, 1), (-3, 1)]
    assert difficulty(ring([list(p) for p in c])) > 0.25


def test_the_ellipse_keeps_the_area():
    p = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
    assert equivalent_ellipse(p).area == __import__("pytest").approx(p.area, rel=0.02)


def test_skill_is_zero_when_the_trivial_answer_already_wins():
    assert skill(0.99, 0.99) == 0.0
    assert skill(0.5, 0.0) == 0.5
