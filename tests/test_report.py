import json
import os

import pytest

from photo2fcstd import report


@pytest.fixture
def run(tmp_path):
    def make(name, rows, sources):
        d = tmp_path / name
        (d / "out").mkdir(parents=True)
        (d / "results.txt").write_text("".join("%s %s %s\n" % r for r in rows))
        (d / "summary.txt").write_text("%s: mean 0.500" % name)
        for part, src in sources.items():
            (d / "out" / (part + ".spec.json")).write_text(
                json.dumps({"mode": "plan", "outline": {"source": "/photos/%s" % src}}))
        return str(d)
    return make


def test_results_skip_failed_builds(run):
    d = run("a", [("00001", "plan", "0.5"), ("00002", "plan", "fail")], {})
    assert report.results(d) == {"00001": ("plan", 0.5)}
    assert report.failures(d) == {"00002": "plan"}


def test_chosen_reads_each_spec_shape():
    assert report.chosen({"outline": {"source": "/p/00001_2.jpg"}}) == "00001_2.jpg"
    assert report.chosen({"revolve": {"source": "/p/00001_3.jpg"}}) == "00001_3.jpg"
    assert report.chosen({"views": {"front": {"source": "/p/00001_1.jpg"}}}) == "00001_1.jpg"
    assert report.chosen(None) is None


def test_rows_carry_the_paired_delta_and_the_view_change(run):
    base = run("base", [("00001", "plan", "0.40"), ("00002", "plan", "0.60")],
               {"00001": "00001_1.jpg", "00002": "00002_1.jpg"})
    cand = run("cand", [("00001", "plan", "0.55"), ("00002", "plan", "0.60")],
               {"00001": "00001_2.jpg", "00002": "00002_1.jpg"})
    rows = {r["part"]: r for r in report.rows_for(cand, base)}
    assert rows["00001"]["delta"] == pytest.approx(0.15)
    assert (rows["00001"]["base_view"], rows["00001"]["view"]) == ("00001_1.jpg", "00001_2.jpg")
    assert rows["00002"]["delta"] == pytest.approx(0.0)
    assert rows["00002"]["base_view"] == rows["00002"]["view"]


def test_failed_parts_appear_with_no_score(run):
    base = run("b2", [("00001", "plan", "0.40")], {"00001": "00001_1.jpg"})
    cand = run("c2", [("00001", "plan", "0.40"), ("00009", "plan", "fail")], {"00001": "00001_1.jpg"})
    rows = {r["part"]: r for r in report.rows_for(cand, base)}
    assert rows["00009"]["failed"] and rows["00009"]["iou"] is None


def test_build_writes_a_page_that_embeds_every_part(run, monkeypatch):
    monkeypatch.setattr(report, "render", lambda run, parts, jobs: {p: False for p in parts})
    base = run("b3", [("00001", "plan", "0.40")], {"00001": "00001_1.jpg"})
    cand = run("c3", [("00001", "plan", "0.55")], {"00001": "00001_2.jpg"})
    out = os.path.join(cand, "report.html")
    path, shown, total = report.build(cand, base, None, 1, out)
    page = open(path).read()
    assert (shown, total) == (0, 1)
    assert "00001_2.jpg" in page and "biggest regression first" in page
    assert "still rendering" in page
    assert json.loads(page.split("const DATA = ")[1].split(", HAS_BASE")[0])[0]["delta"] == pytest.approx(0.15)


def test_a_run_without_a_baseline_offers_no_delta_sort(run, monkeypatch):
    monkeypatch.setattr(report, "render", lambda run, parts, jobs: {p: False for p in parts})
    d = run("solo", [("00001", "plan", "0.40")], {"00001": "00001_1.jpg"})
    out = os.path.join(d, "report.html")
    report.build(d, None, None, 1, out)
    page = open(out).read()
    assert "biggest regression first" not in page and "HAS_BASE = false" in page


def test_the_page_exists_before_any_thumbnail_is_rendered(run, monkeypatch):
    base = run("b4", [("00001", "plan", "0.40")], {"00001": "00001_1.jpg"})
    cand = run("c4", [("00001", "plan", "0.55")], {"00001": "00001_2.jpg"})
    out = os.path.join(cand, "report.html")

    seen = {}

    def slow_render(run_dir, parts, jobs):
        seen["page_existed"] = os.path.exists(out)
        return {p: False for p in parts}

    monkeypatch.setattr(report, "render", slow_render)
    report.build(cand, base, None, 1, out)
    assert seen["page_existed"]
