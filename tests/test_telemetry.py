import json

import numpy as np
import trimesh

from photo2fcstd import telemetry
from photo2fcstd.insight import oracle
from photo2fcstd.verify import consistency, photo_silhouette, rasterise


def test_run_writes_one_event_per_part(tmp_path):
    run = telemetry.Run(str(tmp_path / "events.jsonl"))
    run.record("00001", mode="plan")
    run.record("00001", iou=0.5)
    run.record("00002", mode="stations", iou=0.1)
    path = run.write()
    events = telemetry.load(path)
    assert [e["part"] for e in events] == ["00001", "00002"]
    assert events[0]["mode"] == "plan" and events[0]["iou"] == 0.5


def test_build_event_flags_unconstrained_sketches():
    good = telemetry.build_event({"valid": True, "solids": 1, "sketches": {"a": {"solve": 0, "redundant": [], "conflicting": [], "geometry": 4, "constraints": 8}}})
    bad = telemetry.build_event({"valid": True, "solids": 1, "sketches": {"a": {"solve": -2, "redundant": [3], "conflicting": [], "geometry": 4, "constraints": 9}}})
    assert good["fully_constrained"] and good["constraints"] == 8
    assert not bad["fully_constrained"]


def test_oracle_picks_the_best_mode_per_part():
    per_mode = {"plan": {"a": ("plan", 0.2), "b": ("plan", 0.9)},
                "stations": {"a": ("stations", 0.7), "b": ("stations", 0.1)}}
    out = oracle(per_mode)
    assert out["a"]["best_mode"] == "stations" and out["a"]["best_iou"] == 0.7
    assert out["b"]["best_mode"] == "plan"


def test_rasterise_is_scale_invariant():
    pts = np.random.default_rng(0).normal(size=(500, 2)) * [10, 3]
    a = rasterise(pts)
    b = rasterise(pts * 7.5)
    assert np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1) > 0.9


def test_consistency_prefers_the_matching_shape():
    box = trimesh.creation.box((2, 1, 0.4))
    masks = [photo_silhouette(m) for m in [np.ones((40, 80), bool)]]
    same = consistency(box, [np.ones((40, 80), bool)], n=24)
    other = consistency(trimesh.creation.icosphere(radius=1.0), [np.ones((40, 80), bool)], n=24)
    assert same > other
