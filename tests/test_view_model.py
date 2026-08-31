import numpy as np

from photo2fcstd import modes, view_model


def stats(rect, sol, elong, holes=0):
    return {"rect": rect, "sol": sol, "elong": elong, "ellipse_rms": 0.4,
            "hole_frac": 0.1 * holes, "stroke": 20.0, "nholes": holes, "area": 500.0}


def test_features_are_per_view_and_finite():
    x = view_model.features([stats(0.9, 0.9, 2.0), stats(0.4, 0.7, 9.0), stats(0.6, 0.8, 4.0)])
    assert x.shape[0] == 3
    assert np.isfinite(x).all()
    assert not np.allclose(x[0], x[1])


def test_a_single_view_is_never_second_guessed():
    assert view_model.choose([{"shape": {}, "elongation": 1}]) is None


def test_the_model_is_on_by_default():
    assert modes.USE_VIEW_MODEL


def test_falls_back_to_the_rules_without_a_model(monkeypatch):
    monkeypatch.setattr(view_model, "_CACHE", {"m": None})
    fallback = object()
    assert modes.outline_source([1, 2, 3], fallback) is fallback
