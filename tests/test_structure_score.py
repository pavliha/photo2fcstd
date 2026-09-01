from photo2fcstd.sketch_score import structure_score, verdict


def case(mine, ideal, loops_mine=1, loops_ideal=1, iou=0.0):
    return {"counts_mine": mine, "counts_ideal": ideal, "loops_mine": loops_mine,
            "loops_ideal": loops_ideal, "region_iou": iou}


def test_a_perfect_drawing_scores_one():
    assert structure_score(case({"line": 4}, {"line": 4}))["structure"] == 1.0


def test_smoothing_notches_away_is_penalised_though_iou_is_high():
    s = structure_score(case({"line": 4}, {"line": 12}, iou=0.97))
    assert s["elements"] < 0.4 and s["structure"] < 0.8


def test_a_missed_hole_costs_a_loop():
    assert structure_score(case({"line": 4}, {"line": 4, "circle": 1}, loops_ideal=2))["loops"] == 0.5


def test_arcs_drawn_as_chords_cost_the_curve_term():
    assert structure_score(case({"line": 10}, {"line": 4, "arc": 6}))["curves"] == 0.0


def test_verdict_carries_iou_and_the_discriminating_flag():
    v = verdict({**case({"line": 4}, {"line": 4}, iou=0.8), "trivial": 0.9})
    assert v["region_iou"] == 0.8 and v["exact"] and not v["discriminating"]
