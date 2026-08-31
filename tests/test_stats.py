import numpy as np
import pytest

from photo2fcstd import stats


def test_mean_ci_brackets_the_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(0.4, 0.25, 200)
    mean, lo, hi = stats.mean_ci(v)
    assert lo < mean < hi
    assert hi - lo == pytest.approx(2 * 1.96 * v.std() / np.sqrt(len(v)), rel=0.25)


def test_paired_delta_finds_a_real_shift():
    before = {str(i): 0.4 for i in range(200)}
    after = {str(i): 0.45 for i in range(200)}
    d = stats.paired_delta(before, after)
    assert d["delta"] == pytest.approx(0.05) and d["significant"] and d["n"] == 200


def test_paired_delta_rejects_noise():
    rng = np.random.default_rng(2)
    before = {str(i): float(x) for i, x in enumerate(rng.normal(0.4, 0.2, 200))}
    after = {k: v + float(rng.normal(0, 0.2)) for k, v in before.items()}
    assert not stats.paired_delta(before, after)["significant"]


def test_paired_delta_needs_shared_parts():
    assert stats.paired_delta({"a": 0.1}, {"b": 0.2}) is None


def test_resolution_and_sample_size_scale_together():
    rng = np.random.default_rng(3)
    small = rng.normal(0.4, 0.25, 200)
    large = rng.normal(0.4, 0.25, 800)
    assert stats.resolution(large) < stats.resolution(small)
    assert stats.parts_needed(small, 0.01) > len(small)
    assert stats.parts_needed(small, 0.10) < len(small)


def test_empty_input_does_not_explode():
    assert stats.mean_ci([]) == (0.0, 0.0, 0.0)
    assert stats.parts_needed([], 0.01) == 0


def test_summary_reports_paired_resolution_not_the_mean(tmp_path, monkeypatch):
    import photo2fcstd.bench as bench
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    base = tmp_path / "runs" / "base"
    base.mkdir(parents=True)
    rng = np.random.default_rng(4)
    values = {str(i): float(v) for i, v in enumerate(rng.uniform(0.1, 0.9, 200))}
    (base / "results.txt").write_text("".join("%s plan %.3f\n" % (k, v) for k, v in values.items()))
    scored = [(k, "%.3f" % (v + 0.02), "") for k, v in values.items()]
    line = bench.summarise("new", scored, 10.0, "base")
    assert "paired resolution" in line
    paired = float(line.split("paired resolution +-")[1].split(";")[0])
    spread = float(line.split("[")[1].split(",")[1].split("]")[0]) - float(line.split("[")[1].split(",")[0])
    assert paired < spread
    assert "real" in line


def test_summary_without_a_baseline_says_so(tmp_path, monkeypatch):
    import photo2fcstd.bench as bench
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    line = bench.summarise("solo", [("a", "0.4", ""), ("b", "0.5", "")], 1.0, None)
    assert "no baseline given" in line
