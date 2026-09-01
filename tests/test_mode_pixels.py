import numpy as np

from photo2fcstd import mode_pixels


class Head:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.array([self.value])


def model(**values):
    return {"kind": "pixels", "dims": 4, "heads": {k: Head(v) for k, v in values.items()}}


def test_on_by_default_and_switchable_off(monkeypatch):
    assert mode_pixels.ENABLED
    monkeypatch.setattr(mode_pixels, "ENABLED", False)
    assert mode_pixels.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}], ["plan"]) is None


def test_picks_the_highest_scoring_allowed_mode(monkeypatch):
    from photo2fcstd import embed
    monkeypatch.setattr(mode_pixels, "ENABLED", True)
    monkeypatch.setattr(mode_pixels, "_CACHE", {"m": model(plan=0.4, profile=0.9, revolve=0.95)})
    monkeypatch.setattr(embed, "for_views", lambda views, allow: np.zeros(4))
    views = [{"source": "/p/a_1.jpg", "elongation": 1.0}]
    assert mode_pixels.predict(views, ["plan", "profile"]) == "profile"
    assert mode_pixels.predict(views, ["plan"]) == "plan"


def test_no_vector_means_no_opinion(monkeypatch):
    from photo2fcstd import embed
    monkeypatch.setattr(mode_pixels, "ENABLED", True)
    monkeypatch.setattr(mode_pixels, "_CACHE", {"m": model(plan=0.4)})
    monkeypatch.setattr(embed, "for_views", lambda views, allow: None)
    assert mode_pixels.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}], ["plan"]) is None


def test_a_wrong_sized_vector_is_refused(monkeypatch):
    from photo2fcstd import embed
    monkeypatch.setattr(mode_pixels, "ENABLED", True)
    monkeypatch.setattr(mode_pixels, "_CACHE", {"m": model(plan=0.4)})
    monkeypatch.setattr(embed, "for_views", lambda views, allow: np.zeros(9))
    assert mode_pixels.predict([{"source": "/p/a_1.jpg", "elongation": 1.0}], ["plan"]) is None
