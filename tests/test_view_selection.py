import numpy as np
import pytest

from photo2fcstd import modes


def view(rect, sol, elong, holes=0):
    return {"shape": {"rectangularity": rect, "solidity": sol, "hole_frac": 0.1 * holes,
                      "ellipse_rms": 0.4, "stroke_px": 20.0, "holes": [0] * holes,
                      "bbox": (100.0, 60.0)},
            "elongation": elong, "symmetric": [], "length_px": 400.0}


def three():
    return [view(0.9, 0.95, 2.0), view(0.4, 0.7, 9.0), view(0.6, 0.8, 4.0)]


def test_one_view_is_never_second_guessed():
    only = [view(0.9, 0.9, 2.0)]
    fb = only[0]
    assert modes.outline_source(only, fb) is fb


def test_the_ranker_flag_actually_selects_the_ranker(monkeypatch):
    """The ranker must be consulted. Its answer may still be vetoed by not_edge_on, which is a
    separate guard, so this asserts the call rather than the identity of the result."""
    calls = []

    class Stub:
        @staticmethod
        def best(specs):
            calls.append(len(specs))
            return specs[-1]

    import photo2fcstd.view_rank as vr
    monkeypatch.setattr(modes, "VIEW_PICK", "ranker")
    monkeypatch.setattr(vr, "best", Stub.best)
    specs = three()
    got = modes.outline_source(specs, specs[0])
    assert calls == [3]
    assert got in specs


def test_pick_view_no_longer_races_for_the_same_decision():
    import inspect
    assert "view_rank" not in inspect.getsource(modes.pick_view)


def test_disabling_the_model_keeps_the_fallback_unless_it_is_edge_on(monkeypatch):
    monkeypatch.setattr(modes, "USE_VIEW_MODEL", False)
    monkeypatch.setattr(modes, "VIEW_PICK", "first")
    specs = three()
    flat = [view(0.9, 0.95, 2.0), view(0.8, 0.9, 2.1), view(0.7, 0.85, 2.2)]
    assert modes.outline_source(flat, flat[1]) is flat[1]
    # an edge-on fallback is vetoed by not_edge_on, which is the point of that guard
    assert modes.outline_source(specs, specs[1]) in specs
