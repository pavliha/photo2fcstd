import numpy as np
import pytest

from photo2fcstd import mode_model, modes


class Classifier:
    classes_ = np.array(["plan", "profile", "revolve", "stations"])

    def predict(self, x):
        return np.array(["stations"])

    def predict_proba(self, x):
        return np.array([[0.2, 0.15, 0.05, 0.6]])


def event():
    """The telemetry shape mode_model.features reads."""
    return {"elongation": 1.2, "rectangularity": 0.8, "solidity": 0.95, "hole_frac": 0.0,
            "min_over_max": 0.7, "ellipse_rms": 0.4, "stroke_px": 20.0, "stations": 3,
            "ellipse_aspect": 0.8, "round": 0, "roundish": 0, "length_px": 400.0,
            "symmetric": []}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "m.joblib"
    path.write_text("x")
    monkeypatch.setattr(mode_model, "MODEL_PATH", str(path))
    monkeypatch.setattr(mode_model, "_CACHED", {str(path): Classifier()})
    return path


def test_a_mode_the_caller_forbids_is_never_returned(stub):
    got = mode_model.predict([event()], allowed=["profile", "plan", "revolve"])
    assert got != "stations"
    assert got == "plan"


def test_without_a_list_the_classifier_is_left_alone(stub):
    assert mode_model.predict([event()], allowed=None) == "stations"


def test_learned_mode_never_offers_stations():
    import inspect
    src = inspect.getsource(modes.learned_mode)
    assert '"stations"' not in src and "'stations'" not in src


def test_no_allowed_class_falls_back_rather_than_guessing(stub):
    from photo2fcstd import fallback
    fallback.reset()
    assert mode_model.predict([event()], allowed=["nonesuch"]) is None
    assert any(c == "mode_model" for c, _ in fallback.events())
    fallback.reset()
