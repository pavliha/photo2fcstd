import json

import pytest

from photo2fcstd import bench, sketch_score


def square_spec(size=10.0, depth=2.0):
    pts = [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]]
    elements = [{"type": "line", "p0": pts[i], "p1": pts[(i + 1) % 4]} for i in range(4)]
    return {"name": "sq", "mode": "plan", "mm_per_px": 1.0, "unit": "mm", "views": {}, "revolve": None,
            "outline": {"source": "x.jpg", "loops": [{"type": "loop", "elements": elements,
                                                      "kinds": ["H", "V", "H", "V"], "joins": [""] * 4}],
                        "depth_px": depth, "depth_note": "test"}}


def square_ideal(size=10.0):
    loop = [{"type": "line", "xy": [[0.0, 0.0], [size, 0.0]]}, {"type": "line", "xy": [[size, 0.0], [size, size]]},
            {"type": "line", "xy": [[size, size], [0.0, size]]}, {"type": "line", "xy": [[0.0, size], [0.0, 0.0]]}]
    return {"prism": True, "depth": 2.0, "n_loops": 1, "counts": {"line": 4}, "loops": [loop], "face_area": size * size}


def test_a_matching_sketch_scores_near_one():
    row = sketch_score.score_one(square_spec(), square_ideal())
    assert row["has_sketch"]
    assert row["region_iou"] > 0.95
    assert row["counts_mine"] == row["counts_ideal"]


def test_a_wrong_sketch_scores_low():
    row = sketch_score.score_one(square_spec(size=10.0), square_ideal(size=10.0) | {"loops": [
        [{"type": "circle", "xy": [[5.0 + 5.0 * c, 5.0 + 5.0 * s] for c, s in
                                   [(1, 0), (0, 1), (-1, 0), (0, -1)]]}]], "counts": {"circle": 1}})
    assert row["region_iou"] < 0.95
    assert row["counts_mine"] != row["counts_ideal"]


def test_bench_reads_specs_and_summarises(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "r"
    (run / "out").mkdir(parents=True)
    json.dump(square_spec(), open(run / "out" / "00001.spec.json", "w"))
    ideal = tmp_path / "ideal.json"
    json.dump({"00001": square_ideal()}, open(ideal, "w"))
    monkeypatch.setattr(bench, "IDEAL_SKETCHES", str(ideal))
    scores = bench.sketch_scores(str(run), str(ideal))
    assert set(scores) == {"00001"}
    line = bench.sketch_line(scores)
    assert "1 of 1 parts emit a sketch" in line
    assert "1 reproduce its exact primitives" in line


def test_missing_ideal_file_is_not_fatal(tmp_path):
    (tmp_path / "out").mkdir()
    assert bench.sketch_scores(str(tmp_path), str(tmp_path / "nope.json")) == {}
    assert bench.sketch_line({}) == ""
