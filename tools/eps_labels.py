"""For each part, which simplification tolerance gives the best sketch."""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "src")
from photo2fcstd import bench

GRID = [0.002, 0.004, 0.008, 0.015, 0.025]
OUT = "data/eps_labels.json"


def run_at(eps, ids, jobs):
    name = "eps_%s" % str(eps).replace(".", "")
    run = os.path.join("runs", name)
    shutil.rmtree(run, ignore_errors=True)
    env = dict(os.environ, P2F_RUN_EPS=str(eps), P2F_REPEATED_RUN_EPS=str(eps))
    subprocess.run([os.path.expanduser("~/3DPrint/.venv/bin/photo2fcstd-bench"), name,
                    "--jobs", str(jobs), "--ids", ids, "--sketch-only"],
                   env=env, capture_output=True, check=False)
    out = {}
    for part, row in bench.sketch_scores(run).items():
        value = row.get("primitive_f1")
        if row.get("trustworthy") and isinstance(value, dict) and value.get("wanted"):
            out[part] = float(value["f1"])
    shutil.rmtree(run, ignore_errors=True)
    return out


def main():
    ids = sys.argv[1] if len(sys.argv) > 1 else "data/tune_ids.txt"
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    table = {}
    for eps in GRID:
        scores = run_at(eps, ids, jobs)
        print("eps %.3f: %d parts, mean F1 %.4f"
              % (eps, len(scores), sum(scores.values()) / max(len(scores), 1)), flush=True)
        for part, value in scores.items():
            table.setdefault(part, {})[str(eps)] = value
    labels = {}
    for part, row in table.items():
        if len(row) < len(GRID):
            continue
        best = max(row, key=row.get)
        labels[part] = {"best_eps": float(best), "scores": row,
                        "gain": row[best] - row[str(GRID[3])]}
    json.dump(labels, open(OUT, "w"))
    import numpy as np
    fixed = {str(e): float(np.mean([r["scores"][str(e)] for r in labels.values()])) for e in GRID}
    oracle = float(np.mean([max(r["scores"].values()) for r in labels.values()]))
    print("\n%d parts labelled" % len(labels))
    for e in GRID:
        print("  fixed eps %.3f -> %.4f" % (e, fixed[str(e)]))
    print("  per-part oracle -> %.4f  (headroom %+.4f over the best fixed)"
          % (oracle, oracle - max(fixed.values())))


if __name__ == "__main__":
    main()
