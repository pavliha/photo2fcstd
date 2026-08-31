import numpy as np
import pytest

from photo2fcstd import analysis, modes

EXPECTED = {"01407": "revolve", "00359": "revolve", "01326": "revolve", "01821": "revolve",
            "01289": "plan", "00476": "plan", "01540": "plan", "00308": "plan",
            "00171": "profile", "00133": "profile", "00621": "profile", "00709": "profile",
            "00523": "stations", "01745": "stations", "00201": "stations", "01167": "stations"}
FLOOR_IOU = 0.35


@pytest.fixture(scope="module")
def chosen(dataset, photos_of):
    return {part: modes.select([analysis.view(p) for p in photos_of(part)[:3]])[0] for part in EXPECTED}


def test_every_mode_stays_reachable(chosen):
    assert set(chosen.values()) == {"stations", "profile", "plan", "revolve"}, chosen


def test_known_parts_keep_their_mode(chosen):
    wrong = {p: (chosen[p], want) for p, want in EXPECTED.items() if chosen[p] != want}
    assert not wrong, wrong


def test_mode_mix_is_not_degenerate(chosen):
    counts = {m: sum(1 for v in chosen.values() if v == m) for m in set(chosen.values())}
    assert max(counts.values()) <= len(EXPECTED) * 0.6, counts


@pytest.mark.build
def test_mini_benchmark_stays_above_the_floor(dataset, freecad, tmp_path):
    from photo2fcstd.bench import run_bench
    import photo2fcstd.bench as bench
    bench.ROOT = str(tmp_path)
    summary = run_bench("mini", 4, list(EXPECTED)[:8])
    mean = float(summary.split("mean ")[1].split()[0])
    assert mean >= FLOOR_IOU, summary
