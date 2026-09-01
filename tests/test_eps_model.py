import numpy as np

from photo2fcstd import eps_model, trace


def square(n=60, side=200.0):
    return np.vstack([np.linspace([0, 0], [side, 0], n), np.linspace([side, 0], [side, side], n),
                      np.linspace([side, side], [0, side], n), np.linspace([0, side], [0, 0], n)])


def ring(n=300, r=100.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)]) * r + [200, 200]


def test_it_is_off_by_default():
    assert not eps_model.ENABLED


def test_counts_move_for_a_curve_and_not_for_a_square():
    curved = eps_model.element_counts(ring(), 200.0)
    flat = eps_model.element_counts(square(), 200.0)
    assert max(curved) > min(curved)
    assert max(flat) == min(flat)


def test_the_tolerance_is_restored_after_counting():
    before = trace.RUN_EPS
    eps_model.element_counts(ring(), 200.0)
    assert trace.RUN_EPS == before


def test_one_row_per_candidate():
    c = eps_model.element_counts(ring(), 200.0)
    view = {"shape": {"rectangularity": 0.8, "solidity": 0.95, "hole_frac": 0.0,
                      "ellipse_rms": 0.4, "stroke_px": 20.0, "holes": []},
            "elongation": 1.1, "length_px": 200.0}
    x = eps_model.per_candidate(view, ring(), c)
    assert x.shape[0] == len(eps_model.CANDIDATES)
    assert np.isfinite(x).all()


def test_disabled_means_the_pipeline_keeps_its_constant():
    view = {"shape": {"rectangularity": 0.8, "solidity": 0.95, "hole_frac": 0.0,
                      "ellipse_rms": 0.4, "stroke_px": 20.0, "holes": []},
            "elongation": 1.1, "length_px": 200.0}
    assert eps_model.choose(view, ring()) is None
