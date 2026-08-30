import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import trimesh

from photo2fcstd import cli
from photo2fcstd.score import best_iou
from photo2fcstd.settings import FREECADCMD, data_dir

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def photos_of(part):
    return sorted(glob.glob(os.path.join(data_dir(), "captured_img", "*", part + "_*.jpg")))


def truth_of(part):
    return os.path.join(data_dir(), "stl_from_step", part + ".stl")


def make_spec(args):
    part, out_dir = args
    argv = photos_of(part) + ["--out", os.path.join(out_dir, part + ".FCStd"),
                              "--stl", os.path.join(out_dir, part + ".stl"), "--spec-only"]
    log = os.path.join(out_dir, part + ".log")
    with open(log, "w") as fh:
        out, err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fh
        try:
            doc = cli.main(argv)
            return part, doc["mode"] if "mode" in doc else mode_from_log(log, fh)
        except BaseException as exc:
            fh.write("FAILED %s\n" % exc)
            return part, "none"
        finally:
            sys.stdout, sys.stderr = out, err


def mode_from_log(path, fh=None):
    if fh:
        fh.flush()
    for line in open(path, errors="ignore"):
        if line.startswith("mode: "):
            return line.split()[1]
    return "none"


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


def warm_masks(parts):
    from photo2fcstd.trace import cached_mask, segment_photo
    photos = [p for part in parts for p in photos_of(part)]
    todo = [p for p in photos if not os.path.exists(cached_mask(p))]
    for p in todo:
        segment_photo(p)
    return len(photos), len(todo)


def build_shard(args):
    listing, log = args
    builder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.py")
    with open(log, "w") as fh:
        subprocess.run([FREECADCMD, builder], stdout=fh, stderr=subprocess.STDOUT,
                       env=dict(os.environ, P2F_LIST=listing), timeout=7200)
    return sum(1 for l in open(log, errors="ignore") if l.startswith("BATCH "))


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
    if built != len(jobs):
        raise SystemExit("build incomplete: %d of %d - see %s/build*.log" % (built, len(jobs), run_dir))
    return built


def run_bench(name, jobs, parts):
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(run_dir):
        raise SystemExit("runs/%s exists" % name)
    out_dir = os.path.join(run_dir, "out")
    os.makedirs(out_dir)
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(run_dir, "code"))
    t0 = time.time()
    total, fresh = warm_masks(parts)
    print("stage 1 masks: %d cached, %d segmented in %.0f s" % (total - fresh, fresh, time.time() - t0), flush=True)
    t1 = time.time()
    with Pool(jobs) as pool:
        specs = dict(pool.map(make_spec, [(p, out_dir) for p in parts]))
    print("stage 2 specs: %d in %.0f s" % (sum(1 for v in specs.values() if v != "none"), time.time() - t1), flush=True)
    t2 = time.time()
    print("stage 3 build: %d models in %.0f s" % (build_all(run_dir, parts, jobs), time.time() - t2), flush=True)
    t3 = time.time()
    with Pool(jobs) as pool:
        rows = pool.map(score, [(p, out_dir) for p in parts])
    print("stage 4 score: %.0f s" % (time.time() - t3), flush=True)
    results = os.path.join(run_dir, "results.txt")
    with open(results, "w") as fh:
        for part, value, _ in rows:
            fh.write("%s %s %s\n" % (part, specs.get(part, "none"), value))
    with open(os.path.join(run_dir, "fails.log"), "w") as fh:
        for part, value, why in rows:
            if value == "fail":
                fh.write("%s %s\n" % (part, why))
    ok = [float(v) for _, v, _ in rows if v != "fail"]
    summary = "runs/%s: n=%d mean %.3f fails %d in %.0f s" % (name, len(ok), sum(ok) / max(len(ok), 1), len(rows) - len(ok), time.time() - t0)
    open(os.path.join(run_dir, "summary.txt"), "w").write(summary + "\n")
    print(summary)
    return summary


def main(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd-bench")
    p.add_argument("name")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--ids", default=os.path.join(ROOT, "data", "printcad_test_ids.txt"))
    a = p.parse_args(argv)
    run_bench(a.name, a.jobs, open(a.ids).read().split())


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
