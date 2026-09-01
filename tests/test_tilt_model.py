import numpy as np
import pytest

from photo2fcstd import tilt_model as TM


class Head:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.array([self.value])


def fake(monkeypatch, normal, familiar=True):
    m = {"mu": np.zeros(4), "sd": np.ones(4), "heads": [Head(v) for v in normal],
         "centre": np.array([1.0, 0, 0, 0]), "gate": 1.0 if familiar else -1.0,
         "held_out_mae": 4.8}
    monkeypatch.setattr(TM, "_CACHE", {"m": m})
    monkeypatch.setattr("photo2fcstd.embed.vectors_for", lambda paths: np.zeros((len(paths), 4)))
    return m


def test_a_square_photograph_reads_near_zero(monkeypatch):
    fake(monkeypatch, [0.0, 0.0, 1.0])
    assert TM.tilt_of("a.jpg") == pytest.approx(0.0, abs=1e-6)


def test_a_tilted_photograph_reads_its_angle(monkeypatch):
    fake(monkeypatch, [np.sin(np.radians(20)), 0.0, np.cos(np.radians(20))])
    assert TM.tilt_of("a.jpg") == pytest.approx(20.0, abs=1e-4)


def test_an_unfamiliar_photograph_abstains_rather_than_guessing(monkeypatch):
    fake(monkeypatch, [0.0, 0.0, 1.0], familiar=False)
    assert TM.tilt_of("a.jpg") is None


def test_square_check_asks_for_a_reshoot_above_the_threshold(monkeypatch):
    fake(monkeypatch, [np.sin(np.radians(25)), 0.0, np.cos(np.radians(25))])
    problems, notes = TM.square_check(["a.jpg", "b.jpg"])
    assert problems and "off square" in problems[0]


def test_square_check_says_so_when_it_cannot_check(monkeypatch):
    fake(monkeypatch, [0.0, 0.0, 1.0], familiar=False)
    problems, notes = TM.square_check(["a.jpg"])
    assert not problems and any("not checked" in n or "was not checked" in n for n in notes)
