import pytest


def test_primitive_f1_is_one_when_the_counts_match():
    from photo2fcstd.sketch_score import primitive_f1
    got = primitive_f1({"line": 4, "arc": 2}, {"line": 4, "arc": 2})
    assert got["f1"] == pytest.approx(1.0)


def test_over_drawing_and_under_drawing_cost_the_same():
    from photo2fcstd.sketch_score import primitive_f1
    over = primitive_f1({"line": 8}, {"line": 4})
    under = primitive_f1({"line": 2}, {"line": 4})
    assert over["f1"] == pytest.approx(under["f1"], abs=0.02)
    assert over["precision"] < over["recall"] and under["recall"] < under["precision"]


def test_the_wrong_primitive_type_earns_nothing():
    from photo2fcstd.sketch_score import primitive_f1
    assert primitive_f1({"line": 4}, {"arc": 4})["f1"] == 0.0


def test_an_empty_sketch_scores_zero():
    from photo2fcstd.sketch_score import primitive_f1
    assert primitive_f1({}, {"line": 4})["f1"] == 0.0
    assert primitive_f1({"line": 4}, {})["f1"] == 0.0
