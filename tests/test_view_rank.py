import json

import numpy as np
import pytest

from photo2fcstd import view_rank


def shots(values):
    keys = view_rank.KEYS
    return [{k: float(v) for k, v in zip(keys, row)} for row in values]


def test_features_include_own_and_relative_terms():
    s = shots([[10, 0.5, 0.9, 2.0, 1, 0.0, 0.1, 0.2], [20, 0.8, 0.95, 1.2, 2, 0.1, 0.05, 0.3]])
    f0, f1 = view_rank.features(s, 0), view_rank.features(s, 1)
    assert len(f0) == len(view_rank.KEYS) * 4
    assert f0[:len(view_rank.KEYS)] == [s[0][k] for k in view_rank.KEYS]
    assert f1[len(view_rank.KEYS)] == pytest.approx(1.0)
    assert f0[len(view_rank.KEYS)] == pytest.approx(0.5)


def test_shot_of_reads_a_view_record():
    view = {"shape": {"bbox": (40, 20), "rectangularity": 0.8, "solidity": 0.9, "hole_frac": 0.1,
                      "ellipse_rms": 0.2, "stroke_px": 5.0},
            "elongation": 2.0, "symmetric": ["x"], "length_px": 50.0}
    shot = view_rank.shot_of(view)
    assert shot["area"] == 800 and shot["elongation"] == 2.0
    assert shot["symmetry"] == 1 and shot["stroke"] == pytest.approx(0.1)


def test_a_trained_ranker_prefers_the_better_view(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        good = dict(zip(view_rank.KEYS, [100, 0.9, 0.95, 1.05, 2, 0.0, 0.4, 0.1]))
        poor = dict(zip(view_rank.KEYS, [60, 0.6, 0.8, 2.8, 0, 0.0, 0.3, 0.2]))
        good = {k: v * (1 + rng.normal(0, 0.03)) for k, v in good.items()}
        poor = {k: v * (1 + rng.normal(0, 0.03)) for k, v in poor.items()}
        good["iou"], poor["iou"] = 0.85 + rng.normal(0, 0.03), 0.35 + rng.normal(0, 0.03)
        rows.append({"part": "x", "shots": [poor, good]})
    model = view_rank.train(rows)
    s = [rows[0]["shots"][0], rows[0]["shots"][1]]
    scores = model.predict(view_rank.matrix(s))
    assert scores[1] > scores[0]


def test_ranking_is_skipped_without_a_model(monkeypatch, tmp_path):
    monkeypatch.setattr(view_rank, "MODEL_PATH", str(tmp_path / "absent.joblib"))
    view_rank._CACHED.clear()
    assert view_rank.rank([{}, {}]) is None
    assert view_rank.best([{}, {}]) is None


def test_a_single_view_needs_no_ranking():
    assert view_rank.rank([{"shape": {}}]) is None
