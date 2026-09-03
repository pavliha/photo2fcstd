import pytest

from photo2fcstd import analysis, modes

CASES = [("01407", "revolve"), ("00359", "revolve"), ("01289", "plan"), ("00476", "plan"),
         ("00171", "profile"), ("00133", "profile"), ("00523", "profile"), ("01745", "profile")]


@pytest.mark.parametrize("part,expected", CASES)
def test_mode_selection_is_never_far_worse_than_the_rules(dataset, photos_of, part, expected):
    import json
    labels = {r["part"]: r["iou_per_mode"]
              for r in json.load(open("data/mode_labels_full.json")) if r.get("iou_per_mode")}
    views = [analysis.view(p) for p in photos_of(part)[:3]]
    mode, src = modes.select(views)
    scores = labels.get(part, {})
    got, ruled = scores.get(mode), scores.get(expected)
    if got is not None and ruled is not None:
        assert got > ruled - 0.10, "%s: %s scores %.3f, rules' %s scores %.3f" % (part, mode, got, expected, ruled)
    assert src in views


def test_stations_is_never_auto_selected(dataset, photos_of):
    for part in ("00523", "01745", "01289", "00171"):
        views = [analysis.view(p) for p in photos_of(part)[:3]]
        assert modes.select(views)[0] != "stations"


def test_forced_mode_overrides(dataset, photos_of):
    views = [analysis.view(p) for p in photos_of("01289")[:3]]
    assert modes.select(views, "stations")[0] == "stations"
    assert modes.select(views, "revolve")[0] == "revolve"


def test_select_rejects_an_empty_view_list():
    with pytest.raises(ValueError):
        modes.select([])


def test_same_face_detects_a_plate(dataset, photos_of):
    assert modes.same_face([analysis.view(p) for p in photos_of("01289")[:3]])
    assert not modes.same_face([analysis.view(p) for p in photos_of("00523")[:3]])


def test_learned_selection_is_used_when_enabled(dataset, photos_of, monkeypatch, tmp_path):
    from photo2fcstd import analysis, mode_model
    views = [analysis.view(p) for p in photos_of("01289")[:3]]
    monkeypatch.setattr(modes, "LEARNED", True)
    monkeypatch.setattr(modes, "learned_mode", lambda specs: "revolve")
    assert modes.select(views)[0] != "revolve"
    monkeypatch.setattr(modes, "learned_mode", lambda specs: "profile")
    mode, src = modes.select(views)
    assert mode == "profile" and src in views


def test_source_for_picks_a_view_each_mode_can_use(dataset, photos_of):
    from photo2fcstd import analysis
    views = [analysis.view(p) for p in photos_of("01407")[:3]]
    for mode in ("stations", "profile", "plan", "revolve"):
        assert modes.source_for(mode, views) in views


def test_oracle_depth_is_off_unless_asked(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", False)
    src = {"source": "/p/00002_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note, trusted = modes.outline_depth(src, [], "plan", None)
    assert "oracle" not in note


def test_oracle_depth_uses_the_step_file_ratio(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", True)
    monkeypatch.setattr(modes, "_TRUE_RATIOS", {"00002": 0.25})
    src = {"source": "/p/00002_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note, trusted = modes.outline_depth(src, [], "plan", None)
    assert depth == 25.0 and "oracle" in note


def test_oracle_depth_falls_through_for_an_unknown_part(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", True)
    monkeypatch.setattr(modes, "_TRUE_RATIOS", {"00002": 0.25})
    src = {"source": "/p/99999_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note, trusted = modes.outline_depth(src, [], "plan", None)
    assert "oracle" not in note


def test_a_learned_selector_may_not_route_to_stations(monkeypatch, dataset, photos_of):
    from photo2fcstd import analysis, mode_pixels, modes
    seen = {}
    monkeypatch.setattr(mode_pixels, "predict", lambda specs, allowed: seen.setdefault("allowed", allowed))
    views = [analysis.view(p) for p in photos_of("00523")[:3]]
    modes.learned_mode(views)
    assert "stations" not in seen["allowed"]


def test_the_learned_path_keeps_the_view_model(monkeypatch, dataset, photos_of):
    from photo2fcstd import analysis, mode_pixels, modes
    views = [analysis.view(p) for p in photos_of("00523")[:3]]
    monkeypatch.setattr(mode_pixels, "predict", lambda specs, allowed: "plan")
    monkeypatch.setattr(modes, "LEARNED", True)
    seen = {}
    monkeypatch.setattr(modes, "outline_source", lambda specs, fallback: seen.setdefault("used", fallback))
    modes.select(views)
    assert "used" in seen


def test_revolve_from_the_learned_path_skips_the_outline_view_model(monkeypatch, dataset, photos_of):
    from photo2fcstd import analysis, mode_pixels, modes
    views = [analysis.view(p) for p in photos_of("00008")[:3]]
    monkeypatch.setattr(mode_pixels, "predict", lambda specs, allowed: "revolve")
    monkeypatch.setattr(modes, "LEARNED", True)
    called = []
    monkeypatch.setattr(modes, "outline_source", lambda specs, fallback: called.append(1) or fallback)
    mode, _ = modes.select(views)
    assert mode == "revolve" and not called


def test_an_edge_on_view_is_not_offered_to_the_view_model():
    from photo2fcstd import modes
    specs = [{"elongation": 3.0, "source": "a"}, {"elongation": 2.6, "source": "b"},
             {"elongation": 22.4, "source": "c"}]
    assert [v["source"] for v in modes.face_on(specs)] == ["a", "b"]


def test_uniformly_elongated_views_are_all_kept():
    from photo2fcstd import modes
    specs = [{"elongation": 8.0, "source": "a"}, {"elongation": 9.0, "source": "b"}]
    assert len(modes.face_on(specs)) == 2


def test_selection_statistics_survive_a_change_to_regularisation():
    from photo2fcstd import modes
    view = {"select": {"elongation": 4.0, "hole_frac": 0.2, "rectangularity": 0.8},
            "elongation": 99.0, "shape": {"hole_frac": 0.9, "rectangularity": 0.1}}
    assert modes.stat(view, "elongation") == 4.0
    assert modes.stat(view, "hole_frac") == 0.2
    assert modes.stat(view, "rectangularity") == 0.8


def test_a_view_without_selection_statistics_still_works():
    from photo2fcstd import modes
    view = {"elongation": 4.0, "shape": {"hole_frac": 0.2}}
    assert modes.stat(view, "elongation") == 4.0
    assert modes.stat(view, "hole_frac") == 0.2


def test_the_view_model_prefers_the_pre_regularisation_statistics():
    from photo2fcstd import view_model
    view = {"select": {"rectangularity": 0.8, "solidity": 0.9, "elongation": 2.0, "hole_frac": 0.1,
                       "stroke_px": 5.0, "nholes": 3, "length_px": 100.0},
            "elongation": 99.0, "length_px": 1.0,
            "shape": {"rectangularity": 0.1, "solidity": 0.1, "ellipse_rms": 0.3,
                      "hole_frac": 0.9, "stroke_px": 1.0, "holes": []}}
    got = view_model.stats_of_view(view)
    assert got["rect"] == 0.8 and got["elong"] == 2.0 and got["nholes"] == 3
    assert got["ellipse_rms"] == 0.3


def test_a_guessed_depth_is_marked_untrusted(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", False)
    monkeypatch.setattr(modes, "predicted_depth", lambda src, others: None)
    src = {"source": "/p/00002_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note, trusted = modes.outline_depth(src, [], "plan", None)
    assert "caliper" in note and trusted is False


def test_a_caliper_depth_is_trusted():
    from photo2fcstd import modes
    src = {"source": "/p/00002_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note, trusted = modes.outline_depth(src, [], "plan", 42.0)
    assert depth == 42.0 and trusted is True
