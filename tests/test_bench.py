from photo2fcstd import bench


class FakePool:
    def __init__(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, args):
        return [(a[0], "plan", {}) for a in args]


def stub(monkeypatch, tmp_path, built):
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    monkeypatch.setattr(bench, "Pool", FakePool)
    monkeypatch.setattr(bench, "with_photos", lambda parts: (list(parts), []))
    monkeypatch.setattr(bench, "warm_masks", lambda parts: (len(parts), 0))
    monkeypatch.setattr(bench, "sketch_scores", lambda run_dir, **k: {})
    monkeypatch.setattr(bench, "build_all", lambda *a, **k: built.append(1) or 0)


def test_sketch_only_stops_before_freecad(tmp_path, monkeypatch):
    built = []
    stub(monkeypatch, tmp_path, built)
    summary = bench.run_bench("sk", 1, ["00001"], sketch_only=True)
    assert not built
    assert "no solids built" in summary


def test_sketch_delta_is_paired_against_the_baseline_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    base = tmp_path / "runs" / "old"
    base.mkdir(parents=True)
    theirs = {"a": {"region_iou": 0.40, "trustworthy": True},
              "b": {"region_iou": 0.60, "trustworthy": True}}
    mine = {"a": {"region_iou": 0.50, "trustworthy": True},
            "b": {"region_iou": 0.70, "trustworthy": True}}
    monkeypatch.setattr(bench, "sketch_scores", lambda run_dir, **k: theirs)
    line = bench.sketch_delta("old", mine)
    assert "+0.100" in line and "2 shared parts" in line


def test_sketch_delta_is_silent_without_a_baseline():
    assert bench.sketch_delta(None, {"a": {"region_iou": 0.5, "trustworthy": True}}) == ""


def test_untrustworthy_parts_are_excluded_from_the_sketch_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    base = tmp_path / "runs" / "old"
    base.mkdir(parents=True)
    theirs = {"a": {"region_iou": 0.40, "trustworthy": True},
              "b": {"region_iou": 0.10, "trustworthy": False}}
    monkeypatch.setattr(bench, "sketch_scores", lambda run_dir, **k: theirs)
    mine = {"a": {"region_iou": 0.50, "trustworthy": True},
            "b": {"region_iou": 0.90, "trustworthy": False}}
    assert "1 shared parts" in bench.sketch_delta("old", mine)


def test_flattered_counts_high_scores_with_missing_detail():
    from photo2fcstd import bench
    sketches = {
        "a": {"trustworthy": True, "region_iou": 0.9, "counts_mine": {"line": 1}, "counts_ideal": {"line": 60}},
        "b": {"trustworthy": True, "region_iou": 0.9, "counts_mine": {"line": 8}, "counts_ideal": {"line": 8}},
        "c": {"trustworthy": True, "region_iou": 0.4, "counts_mine": {"line": 1}, "counts_ideal": {"line": 60}},
        "d": {"trustworthy": False, "region_iou": 0.9, "counts_mine": {"line": 1}, "counts_ideal": {"line": 60}},
    }
    assert bench.flattered(sketches) == (1, 2)


def test_flattered_is_empty_without_counts():
    from photo2fcstd import bench
    assert bench.flattered({"a": {"trustworthy": True, "region_iou": 0.9}}) == (0, 0)


def test_the_frozen_split_shares_no_part_and_no_geometry_group():
    import json
    import os
    if not (os.path.exists("data/tune_ids.txt") and os.path.exists("data/test_ids.txt")):
        return
    groups = json.load(open("data/part_groups.json"))
    tune = set(open("data/tune_ids.txt").read().split())
    test = set(open("data/test_ids.txt").read().split())
    assert tune and test and not (tune & test)
    assert not ({groups.get(p, -1) for p in tune} & {groups.get(p, -1) for p in test})
