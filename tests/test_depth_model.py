import pytest


def test_pixel_head_is_used_when_it_has_a_vector(monkeypatch):
    import numpy as np
    from photo2fcstd import depth_model, embed

    class Head:
        def predict(self, x):
            return np.array([np.log(0.25)])

    monkeypatch.setattr(depth_model, "_CACHE", {"p": {"kind": "pixels", "model": Head(), "offset": 0.1, "alpha": 0.2, "dims": 6}})
    monkeypatch.setattr(embed, "for_views", lambda views, allow: np.zeros(6))
    ratio, lo, hi, coverage = depth_model.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}])
    assert ratio == pytest.approx(0.25)
    assert lo < ratio < hi and coverage == pytest.approx(0.8)


def test_pixel_head_falls_back_when_no_vector(monkeypatch):
    from photo2fcstd import depth_model, embed
    monkeypatch.setattr(embed, "for_views", lambda views, allow: None)
    monkeypatch.setattr(depth_model, "_CACHE", {"p": {"kind": "pixels", "model": None, "offset": 0.1, "alpha": 0.2, "dims": 6}, "m": None})
    assert depth_model.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}]) is None


def test_a_wrong_sized_vector_is_refused(monkeypatch):
    import numpy as np
    from photo2fcstd import depth_model, embed
    monkeypatch.setattr(embed, "for_views", lambda views, allow: np.zeros(3))
    monkeypatch.setattr(depth_model, "_CACHE", {"p": {"kind": "pixels", "model": None, "offset": 0.1, "alpha": 0.2, "dims": 6}, "m": None})
    assert depth_model.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}]) is None
