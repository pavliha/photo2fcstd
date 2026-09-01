import argparse
import glob
import os
import json
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import trimesh

from photo2fcstd import cli, sketch_score, stats, telemetry
from photo2fcstd.score import best_iou
from photo2fcstd.settings import FREECADCMD, data_dir

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def photos_of(part):
    return sorted(glob.glob(os.path.join(data_dir(), "captured_img", "*", part + "_*.jpg")))


def truth_of(part):
    return os.path.join(data_dir(), "stl_from_step", part + ".stl")


def make_spec(args):
    part, out_dir, mode = args
    t0 = time.time()
    argv = photos_of(part) + ["--out", os.path.join(out_dir, part + ".FCStd"),
                              "--stl", os.path.join(out_dir, part + ".stl"), "--spec-only"]
    if mode:
        argv += ["--mode", mode]
    log = os.path.join(out_dir, part + ".log")
    with open(log, "w") as fh:
        out, err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fh
        try:
            from photo2fcstd import analysis, spec as spec_mod
            views = [analysis.view(p) for p in photos_of(part)[:3]]
            doc = spec_mod.assemble(views, name=part, mode=mode,
                                    stl=os.path.join(out_dir, part + ".stl"), log=lambda *a: None)
            spec_path = os.path.join(out_dir, part + ".spec.json")
            with open(spec_path, "w") as out_fh:
                import json
                json.dump(doc, out_fh)
            chosen = doc["mode"]
            event = {"mode": chosen, "views": [telemetry.view_event(v) for v in views],
                     "spec": telemetry.spec_event(doc), "spec_ms": round(1000 * (time.time() - t0))}
            return part, chosen, event
        except Exception as exc:
            fh.write("FAILED %s\n" % exc)
            return part, "none", {"mode": "none", "error": str(exc)[:200]}
        finally:
            sys.stdout, sys.stderr = out, err


def score(args):
    part, out_dir = args
    stl = os.path.join(out_dir, part + ".stl")
    if not os.path.exists(stl):
        return part, "fail", "no stl"
    try:
        value, _ = best_iou(truth_of(part), trimesh.load(stl))
        return part, "%.3f" % value, ""
    except Exception as exc:
        return part, "fail", str(exc)[:150]


def build_reports(run_dir):
    import json
    out = {}
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith("build") or not name.endswith(".log"):
            continue
        for line in open(os.path.join(run_dir, name), errors="ignore"):
            if line.startswith("BATCH "):
                _, part, payload = line.split(" ", 2)
                try:
                    report = json.loads(payload)
                except ValueError:
                    report = {}
                out[part] = report or {}
    return out


def warm_masks(parts):
    from photo2fcstd.trace import cached_mask, segment_photo
    photos = [p for part in parts for p in photos_of(part)]
    todo = [p for p in photos if not os.path.exists(cached_mask(p))]
    for p in todo:
        segment_photo(p)
    return len(photos), len(todo)


PART_BUDGET_S = float(os.environ.get("P2F_PART_BUDGET", 45))


def completed_parts(log):
    if not os.path.exists(log):
        return []
    return [l.split(" ", 2)[1] for l in open(log, errors="ignore") if l.startswith("BATCH ")]


def build_shard(args):
    listing, log = args
    jobs = [l.rstrip("\n").split("\t") for l in open(listing) if l.strip()]
    builder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.py")
    skipped = []
    built = 0
    while jobs:
        with open(listing, "w") as fh:
            for spec_path, out in jobs:
                fh.write("%s\t%s\n" % (spec_path, out))
        attempt_log = log + ".part"
        timed_out = False
        with open(attempt_log, "w") as fh:
            try:
                subprocess.run([FREECADCMD, builder], stdout=fh, stderr=subprocess.STDOUT,
                               env=dict(os.environ, P2F_LIST=listing), timeout=PART_BUDGET_S * len(jobs) + 60)
            except subprocess.TimeoutExpired:
                timed_out = True
        done = completed_parts(attempt_log)
        with open(log, "a") as out_fh, open(attempt_log, errors="ignore") as in_fh:
            out_fh.write(in_fh.read())
        os.remove(attempt_log)
        built += len(done)
        remaining = [j for j in jobs if os.path.splitext(os.path.basename(j[1]))[0] not in done]
        if not remaining:
            jobs = []
            continue
        skipped.append(os.path.splitext(os.path.basename(remaining[0][1]))[0])
        jobs = remaining[1:]
    if skipped:
        with open(log + ".skipped", "w") as fh:
            fh.write("\n".join(skipped) + "\n")
    return built


def build_all(run_dir, parts, shards):
    out_dir = os.path.join(run_dir, "out")
    jobs = [(os.path.join(out_dir, p + ".spec.json"), os.path.join(out_dir, p + ".FCStd")) for p in parts]
    jobs = [(s, o) for s, o in jobs if os.path.exists(s)]
    shards = max(1, min(shards, len(jobs)))
    listings = []
    for i in range(shards):
        listing = os.path.join(run_dir, "list%d.txt" % i)
        with open(listing, "w") as fh:
            for s, o in jobs[i::shards]:
                fh.write("%s\t%s\n" % (s, o))
        listings.append((listing, os.path.join(run_dir, "build%d.log" % i)))
    with Pool(shards) as pool:
        built = sum(pool.map(build_shard, listings))
    skipped = [p for f in sorted(os.listdir(run_dir)) if f.endswith(".skipped")
               for p in open(os.path.join(run_dir, f)).read().split()]
    if skipped:
        print("  skipped %d parts whose build hung: %s" % (len(skipped), " ".join(skipped[:8])), flush=True)
    if not built:
        raise SystemExit("no models built - see %s/build*.log" % run_dir)
    if built + len(skipped) < len(jobs):
        print("  WARNING: %d built, %d skipped, %d unaccounted of %d" % (built, len(skipped), len(jobs) - built - len(skipped), len(jobs)), flush=True)
    return built


def with_photos(parts):
    have = [p for p in parts if photos_of(p)]
    missing = [p for p in parts if not photos_of(p)]
    return have, missing


def scores_of(run):
    path = os.path.join(ROOT, "runs", run, "results.txt")
    if not os.path.exists(path):
        return {}
    return {l.split()[0]: float(l.split()[2]) for l in open(path)
            if len(l.split()) == 3 and l.split()[2] != "fail"}


IDEAL_SKETCHES = os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")


def sketch_scores(run_dir, ideal_path=IDEAL_SKETCHES):
    if not os.path.exists(ideal_path):
        return {}
    ideal = json.load(open(ideal_path))
    out = {}
    for name in sorted(os.listdir(os.path.join(run_dir, "out"))):
        if not name.endswith(".spec.json"):
            continue
        part = name.split(".")[0]
        record = ideal.get(part)
        if not record or "loops" not in record:
            continue
        try:
            out[part] = sketch_score.score_one(json.load(open(os.path.join(run_dir, "out", name))), record)
        except Exception:
            continue
    return out


def sketch_delta(baseline, sketches):
    if not baseline:
        return ""
    path = baseline if os.path.isdir(baseline) else os.path.join(ROOT, "runs", baseline)
    if not os.path.isdir(path):
        return ""
    theirs = sketch_scores(path)
    pick = lambda rows: {k: float(v["region_iou"]) for k, v in rows.items()
                         if v.get("trustworthy") and v.get("region_iou") is not None}
    before, after = pick(theirs), pick(sketches)
    got = stats.paired_delta(before, after)
    if not got:
        return ""
    return ("\n  sketch vs %s: %+.3f [%+.3f, %+.3f] on %d shared parts%s"
            % (os.path.basename(path.rstrip("/")), got["delta"], got["lo"], got["hi"], got["n"],
               "" if got["significant"] else " - indistinguishable from zero"))


def flattered(sketches, score_min=0.8, drawn_max=0.5):
    """Parts scoring well while drawing a fraction of the primitives the part has."""
    hit = total = 0
    for r in sketches.values():
        if not r.get("trustworthy") or r.get("region_iou") is None:
            continue
        mine = sum(r.get("counts_mine", {}).values()) if isinstance(r.get("counts_mine"), dict) else 0
        ideal = sum(r.get("counts_ideal", {}).values()) if isinstance(r.get("counts_ideal"), dict) else 0
        if not mine or not ideal or r["region_iou"] <= score_min:
            continue
        total += 1
        hit += int(mine / ideal < drawn_max)
    return hit, total


def sketch_line(sketches):
    if not sketches:
        return ""
    drawn = [r for r in sketches.values() if r["has_sketch"]]
    if not drawn:
        return ("\n  sketch comparison does not apply to this run: no part produced an outline or a "
                "revolve profile (a stations model has no single face to compare)")
    trusted = [r for r in drawn if r["trustworthy"]]
    exact = [r for r in trusted if r["counts_mine"] == r["counts_ideal"]]
    line = "\n  sketch vs the ideal one: applies to %d of %d parts (outline and revolve modes)" % (
        len(drawn), len(sketches))
    if not trusted:
        return line + "; none of them has trustworthy ground truth to compare against"
    mean, lo, hi = stats.mean_ci([r["region_iou"] for r in trusted])
    line += "; region IoU %.3f [%.3f, %.3f] on the %d with trustworthy truth; %d reproduce its exact primitives" % (
        mean, lo, hi, len(trusted), len(exact))
    keen = [r for r in trusted if r.get("discriminating")]
    if not keen:
        return line
    triv = sum(r["trivial"] for r in keen) / len(keen)
    got = sum(r["region_iou"] for r in keen) / len(keen)
    scored = [r["primitive_f1"] for r in sketches.values()
              if r.get("trustworthy") and isinstance(r.get("primitive_f1"), dict) and r["primitive_f1"]["wanted"]]
    if scored:
        mean, lo, hi = stats.mean_ci([r["f1"] for r in scored])
        line += ("\n  primitive F1 %.3f [%.3f, %.3f] on %d parts (precision %.3f, recall %.3f) - "
                 "drawing too much and too little cost the same here"
                 % (mean, lo, hi, len(scored),
                    sum(r["precision"] for r in scored) / len(scored),
                    sum(r["recall"] for r in scored) / len(scored)))
    hit, total = flattered(sketches)
    if total:
        line += ("\n  of the %d parts scoring over 0.8, %d (%.0f%%) draw under half the primitives "
                 "the part has - a high score there is the outline agreeing, not the features"
                 % (total, hit, 100.0 * hit / total))
    return line + ("\n  against the baseline: drawing one circle and ignoring the photo scores %.3f "
                   "on the %d parts that can tell the difference, where this run scores %.3f (skill %.3f)"
                   % (triv, len(keen), got, (got - triv) / max(1.0 - triv, 1e-9)))


def summarise(name, scored, elapsed, baseline=None, sketches=None):
    ok = {part: float(v) for part, v, _ in scored if v != "fail"}
    mean, lo, hi = stats.mean_ci(ok.values())
    line = "runs/%s: n=%d mean %.3f [%.3f, %.3f] fails %d in %.0f s" % (
        name, len(ok), mean, lo, hi, len(scored) - len(ok), elapsed)
    delta = stats.paired_delta(scores_of(baseline), ok) if baseline else None
    if baseline and delta and delta["lo"] == delta["hi"] == 0.0 and delta["delta"] == 0.0:
        line += "\n  vs %s: identical on all %d shared parts" % (baseline, delta["n"])
    elif baseline and delta:
        half = (delta["hi"] - delta["lo"]) / 2
        line += "\n  vs %s: %+.3f [%+.3f, %+.3f] on %d shared parts - %s" % (
            baseline, delta["delta"], delta["lo"], delta["hi"], delta["n"],
            "real" if delta["significant"] else "indistinguishable from zero")
        line += "\n  paired resolution +-%.3f; detecting +0.01 would need ~%d parts" % (
            half, int(round(delta["n"] * (half / 0.01) ** 2)))
    elif baseline:
        line += "\n  vs %s: no shared parts to compare" % baseline
    else:
        line += "\n  no baseline given; the mean alone resolves +-%.3f" % ((hi - lo) / 2)
    return line + sketch_line(sketches or {})


def run_bench(name, jobs, parts, mode=None, baseline=None, sketch_only=False):
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        raise SystemExit("runs/%s exists" % name)
    out_dir = os.path.join(run_dir, "out")
    os.makedirs(out_dir)
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(run_dir, "code"))
    t0 = time.time()
    parts, missing = with_photos(parts)
    if missing:
        open(os.path.join(run_dir, "no_photos.txt"), "w").write("\n".join(missing) + "\n")
        print("skipping %d parts with no photos (listed in no_photos.txt)" % len(missing), flush=True)
    total, fresh = warm_masks(parts)
    print("stage 1 masks: %d cached, %d segmented in %.0f s" % (total - fresh, fresh, time.time() - t0), flush=True)
    t1 = time.time()
    with Pool(jobs) as pool:
        rows = pool.map(make_spec, [(p, out_dir, mode) for p in parts])
    specs = {p: m for p, m, _ in rows}
    events = telemetry.Run(os.path.join(run_dir, "events.jsonl"))
    for part, _, event in rows:
        events.record(part, **event)
    print("stage 2 specs: %d in %.0f s" % (sum(1 for v in specs.values() if v != "none"), time.time() - t1), flush=True)
    if sketch_only:
        events.write()
        sketches = sketch_scores(run_dir)
        for part, row in sketches.items():
            events.record(part, sketch=row)
        events.write()
        summary = "runs/%s: %d specs in %.0f s, no solids built\n  %s" % (
            name, sum(1 for v in specs.values() if v != "none"), time.time() - t0,
            sketch_line(sketches) + sketch_delta(baseline, sketches))
        open(os.path.join(run_dir, "summary.txt"), "w").write(summary + "\n")
        print(summary)
        return summary
    t2 = time.time()
    print("stage 3 build: %d models in %.0f s" % (build_all(run_dir, parts, jobs), time.time() - t2), flush=True)
    for part, report in build_reports(run_dir).items():
        events.record(part, build=telemetry.build_event(report))
    t3 = time.time()
    with Pool(jobs) as pool:
        scored = pool.map(score, [(p, out_dir) for p in parts])
    print("stage 4 score: %.0f s" % (time.time() - t3), flush=True)
    results = os.path.join(run_dir, "results.txt")
    with open(results, "w") as fh:
        for part, value, _ in scored:
            fh.write("%s %s %s\n" % (part, specs.get(part, "none"), value))
    with open(os.path.join(run_dir, "fails.log"), "w") as fh:
        for part, value, why in scored:
            if value == "fail":
                fh.write("%s %s\n" % (part, why))
    for part, value, why in scored:
        events.record(part, iou=None if value == "fail" else float(value), score_error=why or None)
    events.write()
    sketches = sketch_scores(run_dir)
    for part, row in sketches.items():
        events.record(part, sketch=row)
    summary = summarise(name, scored, time.time() - t0, baseline, sketches)
    open(os.path.join(run_dir, "summary.txt"), "w").write(summary + "\n")
    print(summary)
    return summary


def main(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd-bench")
    p.add_argument("name")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--ids", default=os.path.join(ROOT, "data", "printcad_test_ids.txt"))
    p.add_argument("--mode", choices=("stations", "profile", "plan", "revolve"),
                   help="force every part through one modelling mode")
    p.add_argument("--baseline", help="an earlier run to compare against, paired per part")
    p.add_argument("--sketch-only", action="store_true",
                   help="stop after the sketches; skip FreeCAD and the solid score")
    a = p.parse_args(argv)
    run_bench(a.name, a.jobs, open(a.ids).read().split(), a.mode, a.baseline, a.sketch_only)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
