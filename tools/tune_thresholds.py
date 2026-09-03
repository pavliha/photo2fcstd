"""Random search over the hand-picked thresholds, scored by primitive F1 on the tuning set."""
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "src")
from photo2fcstd import bench, stats

SPACE = {
    "P2F_RUN_EPS": (0.006, 0.030),
    "P2F_REPEATED_RUN_EPS": (0.002, 0.012),
    "P2F_CIRCLE_VETO_AMPLITUDE": (0.04, 0.30),
    "P2F_PERIOD_PROMINENCE": (0.006, 0.030),
    "P2F_PERIOD_AMPLITUDE": (0.015, 0.080),
    "P2F_ANGLE_TOL": (2.0, 9.0),
    "P2F_EDGE_ON_RATIO": (2.0, 6.0),
}
INTEGER = {"P2F_PERIOD_MIN_PEAKS": (4, 10)}
RESULTS = "data/threshold_search.json"


def sample(rng):
    picked = {k: round(rng.uniform(*v), 5) for k, v in SPACE.items()}
    picked.update({k: rng.randint(*v) for k, v in INTEGER.items()})
    return picked


def scorable(ids):
    """Every part in the set with trustworthy truth - the denominator never moves."""
    import json
    from photo2fcstd import sketch_score as SS
    ideal = json.load(open("data/printcad_ideal_sketches_all.json"))
    parts = open(ids).read().split()
    return {p for p in parts if p in ideal and SS.trustworthy(ideal[p])
            and sum((ideal[p].get("counts") or {}).values()) > 0}


def score(name, ids, config, jobs, denominator):
    env = dict(os.environ, **{k: str(v) for k, v in config.items()})
    run = os.path.join("runs", name)
    shutil.rmtree(run, ignore_errors=True)
    subprocess.run([os.path.join(os.path.dirname(sys.executable), "photo2fcstd-bench"), name,
                    "--jobs", str(jobs), "--ids", ids, "--sketch-only"],
                   env=env, capture_output=True, check=False)
    got = bench.sketch_scores(run)
    total = 0.0
    for part in denominator:
        row = got.get(part) or {}
        value = row.get("primitive_f1")
        total += float(value["f1"]) if isinstance(value, dict) and value.get("wanted") else 0.0
    shutil.rmtree(run, ignore_errors=True)
    return total / max(len(denominator), 1), len(got)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ids = sys.argv[2] if len(sys.argv) > 2 else "data/tune_subset.txt"
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    rng = random.Random(7)
    history = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    denominator = scorable(ids)
    base, n = score("tune_base", ids, {}, jobs, denominator)
    print("defaults: primitive F1 %.4f over a fixed %d parts (%d produced a sketch)"
          % (base, len(denominator), n), flush=True)
    for i in range(trials):
        config = sample(rng)
        started = time.time()
        value, count = score("tune_try", ids, config, jobs, denominator)
        history.append({"config": config, "f1": value, "parts": count, "denominator": len(denominator)})
        json.dump(history, open(RESULTS, "w"), indent=1)
        best = max(history, key=lambda r: r["f1"])
        print("%3d/%d  F1 %.4f  best %.4f  (%.0f s)" % (i + 1, trials, value, best["f1"], time.time() - started), flush=True)
    best = max(history, key=lambda r: r["f1"])
    print("\nbest F1 %.4f against defaults %.4f" % (best["f1"], base))
    for k, v in sorted(best["config"].items()):
        print("  %-28s %s" % (k, v))


if __name__ == "__main__":
    main()
