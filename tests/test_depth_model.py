import pytest


def test_pixel_head_is_used_when_enabled_and_it_has_a_vector(monkeypatch):
    import numpy as np
    from photo2fcstd import depth_model, embed
    monkeypatch.setattr(depth_model, "USE_PIXELS", True)

    class Head:
        def predict(self, x):
            return np.array([np.log(0.25)])

    monkeypatch.setattr(depth_model, "_CACHE", {"p": {"kind": "pixels", "model": Head(), "offset": 0.1, "alpha": 0.2, "dims": 6}})
    monkeypatch.setattr(embed, "for_views", lambda views, allow: np.zeros(6))
    ratio, lo, hi, coverage, per_part = depth_model.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}])
    assert ratio == pytest.approx(0.25)
    assert lo < ratio < hi and coverage == pytest.approx(0.8)
    assert per_part is False, "the pixel head has no quantile heads, so its band is the same for every part"


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


def test_pixel_head_is_on_by_default_and_switchable_off(monkeypatch):
    import numpy as np
    from photo2fcstd import depth_model, embed
    assert depth_model.USE_PIXELS
    called = []
    monkeypatch.setattr(depth_model, "USE_PIXELS", False)
    monkeypatch.setattr(embed, "for_views", lambda views, allow: called.append(1) or np.zeros(6))
    monkeypatch.setattr(depth_model, "_CACHE", {"m": None})
    depth_model.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}])
    assert not called


def test_the_note_does_not_claim_a_constant_band_is_this_part_s_uncertainty():
    from photo2fcstd import modes
    src = {"length_px": 100.0, "shape": {"bbox": [10, 10], "stroke_px": 5}, "source": "/p/a_1.jpg"}

    def fixed(views):
        return 0.5, 0.3, 0.9, 0.8, False

    def per_part(views):
        return 0.5, 0.3, 0.9, 0.8, True

    from photo2fcstd import depth_model
    keep = depth_model.predict
    try:
        depth_model.predict = fixed
        _, note = modes.predicted_depth(src, [])
        assert "fixed calibration" in note and "not this part" in note
        depth_model.predict = per_part
        _, note = modes.predicted_depth(src, [])
        assert "of the time between" in note and "fixed calibration" not in note
    finally:
        depth_model.predict = keep
