import numpy as np
import pytest

from photo2fcstd import axis_model, fallback, view_model


@pytest.fixture(autouse=True)
def clean():
    fallback.reset()
    yield
    fallback.reset()


def plate():
    g = np.mgrid[0:30, 0:20, 0:5]
    return {"points_mm": np.column_stack([a.ravel() for a in g]).astype(float), "voxel_mm": 1.0}


def test_a_missing_model_is_announced_once(monkeypatch, capsys):
    monkeypatch.setattr(axis_model, "MODEL_PATH", "/nowhere/axis.joblib")
    monkeypatch.setattr(axis_model, "_CACHE", {})
    assert axis_model.predict_axis(plate()) is None
    assert "axis_model unavailable" in capsys.readouterr().err
    monkeypatch.setattr(axis_model, "_CACHE", {})
    axis_model.predict_axis(plate())
    assert capsys.readouterr().err == ""


def test_the_reason_is_recorded_for_every_occurrence(monkeypatch):
    monkeypatch.setattr(view_model, "MODEL_PATH", "1")
    monkeypatch.setattr(view_model, "_CACHE", {})
    assert view_model.load() is None
    monkeypatch.setattr(view_model, "_CACHE", {})
    assert view_model.load() is None
    got = fallback.events()
    assert len(got) == 2
    assert all(c == "view_model" and "no model at 1" in r for c, r in got)


def test_the_env_flag_that_broke_the_ab_is_now_visible(monkeypatch, capsys):
    monkeypatch.setattr(view_model, "MODEL_PATH", "1")
    monkeypatch.setattr(view_model, "_CACHE", {})
    view_model.load()
    err = capsys.readouterr().err
    assert "view_model unavailable" in err and "no model at 1" in err


def test_quiet_mode_still_records(monkeypatch, capsys):
    monkeypatch.setattr(fallback, "SILENT", True)
    fallback.note("thing", "why")
    assert capsys.readouterr().err == ""
    assert fallback.events() == [("thing", "why")]
