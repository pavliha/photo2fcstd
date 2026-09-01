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
    """Without a model the rule's choice stands, subject to the edge-on veto, so this uses views
    that are all face-on and therefore never vetoed."""
    monkeypatch.setattr(view_model, "_CACHE", {"m": None})
    flat = [{"shape": {"rectangularity": r, "solidity": 0.9, "hole_frac": 0.0, "ellipse_rms": 0.4,
                       "stroke_px": 20.0, "holes": [], "bbox": (100.0, 60.0)},
             "elongation": e, "symmetric": [], "length_px": 400.0}
            for r, e in ((0.9, 2.0), (0.8, 2.1), (0.7, 2.2))]
    assert modes.outline_source(flat, flat[1]) is flat[1]
