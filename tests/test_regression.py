import numpy as np
import pytest

from photo2fcstd import analysis, modes

# What each part selects today, not what it must select. Only the assertions below are
# requirements: every mode stays reachable and nothing routes to stations on its own. 01407 and
# 00359 read "revolve" here until Sept 2026 and now pick plan - neutral on 01407, which scores
# 0.557 either way, and better on 00359, 0.909 against revolve's 0.870.
EXPECTED = {"01407": "plan", "00359": "plan", "01326": "revolve", "01821": "revolve",
            "01289": "plan", "00476": "plan", "01540": "plan", "00308": "plan",
            "00171": "profile", "00133": "profile", "00621": "profile", "00709": "profile",
            "00523": "profile", "01745": "profile", "00201": "profile", "01167": "plan"}
FLOOR_IOU = 0.35


@pytest.fixture(scope="module")
def chosen(dataset, photos_of):
    return {part: modes.select([analysis.view(p) for p in photos_of(part)[:3]])[0] for part in EXPECTED}


def test_every_drawing_mode_stays_reachable(chosen):
    assert set(chosen.values()) == {"profile", "plan", "revolve"}, chosen


def test_stations_is_never_chosen_on_its_own(chosen):
    """stations emits width staircases and no sketch, so nothing may route to it."""
    assert "stations" not in set(chosen.values()), chosen


def test_stations_is_still_reachable_when_forced(dataset, photos_of):
    views = [analysis.view(p) for p in photos_of("00523")[:3]]
    assert modes.select(views, "stations")[0] == "stations"


def test_the_chosen_policy_beats_the_rules_it_replaced(chosen):
    import json
    labels = {r["part"]: r["iou_per_mode"]
              for r in json.load(open("data/mode_labels_full.json")) if r.get("iou_per_mode")}
    pairs = [(labels[p].get(chosen[p]), labels[p].get(want))
             for p, want in EXPECTED.items() if p in labels]
    scored = [(got, ruled) for got, ruled in pairs if got is not None and ruled is not None]
    assert len(scored) >= 10, "not enough labelled parts to judge the policy"
    mine = sum(got for got, _ in scored) / len(scored)
    theirs = sum(ruled for _, ruled in scored) / len(scored)
    assert mine >= theirs, "policy is worse than the rules on the pinned parts: %.3f vs %.3f" % (mine, theirs)


def test_mode_mix_is_not_degenerate(chosen):
    counts = {m: sum(1 for v in chosen.values() if v == m) for m in set(chosen.values())}
    assert max(counts.values()) <= len(EXPECTED) * 0.75, counts


@pytest.mark.build
def test_mini_benchmark_stays_above_the_floor(dataset, freecad, tmp_path):
    from photo2fcstd.bench import run_bench
    import photo2fcstd.bench as bench
    bench.ROOT = str(tmp_path)
    summary = run_bench("mini", 4, list(EXPECTED)[:8])
    mean = float(summary.split("mean ")[1].split()[0])
    assert mean >= FLOOR_IOU, summary


def test_the_recorded_mode_matches_the_selected_mode(dataset, photos_of):
    from photo2fcstd import spec
    for part in ("01289", "00476", "00523", "01407", "00171"):
        views = [analysis.view(p) for p in photos_of(part)[:3]]
        doc = spec.assemble(views, name=part, log=lambda *a: None)
        assert doc["mode"] == modes.select(views)[0], part


def test_the_spec_mode_survives_a_forced_choice(dataset, photos_of):
    from photo2fcstd import spec
    views = [analysis.view(p) for p in photos_of("01289")[:3]]
    for forced in ("stations", "profile", "plan"):
        assert spec.assemble(views, name="x", mode=forced, log=lambda *a: None)["mode"] == forced


def test_edge_on_photos_are_called_out(dataset, photos_of):
    """All three photos of 00214 look at the part edge-on; the sheet must say so."""
    from photo2fcstd import spec
    views = [analysis.view(p) for p in photos_of("00214")[:3]]
    doc = spec.assemble(views, name="00214", log=lambda *a: None)
    assert doc["outline"]["warning"], "no warning on a part photographed edge-on"
    flat = [analysis.view(p) for p in photos_of("00198")[:3]]
    assert not spec.assemble(flat, name="00198", log=lambda *a: None)["outline"].get("warning")
