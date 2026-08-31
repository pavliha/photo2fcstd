import pytest

from photo2fcstd import analysis, modes

CASES = [("01407", "revolve"), ("00359", "revolve"), ("01289", "plan"), ("00476", "plan"),
         ("00171", "profile"), ("00133", "profile"), ("00523", "profile"), ("01745", "profile")]


@pytest.mark.parametrize("part,expected", CASES)
def test_mode_selection(dataset, photos_of, part, expected):
    views = [analysis.view(p) for p in photos_of(part)[:3]]
    mode, src = modes.select(views)
    assert mode == expected
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
    depth, note = modes.outline_depth(src, [], "plan", None)
    assert "oracle" not in note


def test_oracle_depth_uses_the_step_file_ratio(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", True)
    monkeypatch.setattr(modes, "_TRUE_RATIOS", {"00002": 0.25})
    src = {"source": "/p/00002_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note = modes.outline_depth(src, [], "plan", None)
    assert depth == 25.0 and "oracle" in note


def test_oracle_depth_falls_through_for_an_unknown_part(monkeypatch):
    from photo2fcstd import modes
    monkeypatch.setattr(modes, "ORACLE_DEPTH", True)
    monkeypatch.setattr(modes, "_TRUE_RATIOS", {"00002": 0.25})
    src = {"source": "/p/99999_1.jpg", "length_px": 100.0, "shape": {"bbox": (10, 20), "stroke_px": 3.0}}
    depth, note = modes.outline_depth(src, [], "plan", None)
    assert "oracle" not in note
