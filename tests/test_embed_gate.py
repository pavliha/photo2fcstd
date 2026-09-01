import numpy as np
import pytest

from photo2fcstd import depth_model


def centre():
    d = dict(np.load(depth_model.CENTRE_PATH))
    return np.asarray(d["centre"], float), float(d["p95"])


def at_distance(c, want):
    """A unit vector whose cosine distance to c is `want`."""
    rng = np.random.default_rng(0)
    perp = rng.standard_normal(len(c))
    perp -= (perp @ c) * c
    perp /= np.linalg.norm(perp)
    cos = 1.0 - want
    v = cos * c + np.sqrt(max(1 - cos ** 2, 0.0)) * perp
    return v / np.linalg.norm(v)


def test_a_familiar_embedding_is_admitted():
    c, p95 = centre()
    assert depth_model.embedding_is_familiar(np.concatenate([at_distance(c, p95 * 0.5)] * 3))


def test_a_tless_like_embedding_is_refused():
    c, _ = centre()
    assert not depth_model.embedding_is_familiar(np.concatenate([at_distance(c, 0.839)] * 3))


def test_the_gate_can_be_turned_off(monkeypatch):
    c, _ = centre()
    monkeypatch.setattr(depth_model, "GATE", False)
    assert depth_model.embedding_is_familiar(np.concatenate([at_distance(c, 0.95)] * 3))


def test_refusing_is_announced_and_falls_back(monkeypatch):
    from photo2fcstd import fallback
    fallback.reset()
    c, _ = centre()
    far = np.concatenate([at_distance(c, 0.9)] * 3)
    assert not depth_model.embedding_is_familiar(far)
    depth_model._give_up("test")
    assert any(comp == "depth_model" for comp, _ in fallback.events())
    fallback.reset()
