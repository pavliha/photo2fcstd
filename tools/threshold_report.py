"""Which of the hand-picked thresholds actually matter, from the search history."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import stats

HISTORY = "data/threshold_search.json"


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    history = json.load(open(HISTORY))
    scores = np.array([r["f1"] for r in history], float)
    keys = sorted(history[0]["config"])
    print("%d trials, F1 %.4f to %.4f (median %.4f)"
          % (len(history), scores.min(), scores.max(), float(np.median(scores))))
    order = np.argsort(scores)
    top = set(order[-max(3, len(history) // 5):].tolist())
    print("\n%-28s %8s %10s %10s" % ("threshold", "rho", "best fifth", "worst fifth"))
    rows = []
    for key in keys:
        values = np.array([r["config"][key] for r in history], float)
        rho = spearman(values, scores)
        hi = float(np.mean([values[i] for i in range(len(values)) if i in top]))
        lo = float(np.mean([values[i] for i in range(len(values)) if i in set(order[:max(3, len(history) // 5)].tolist())]))
        rows.append((abs(rho), key, rho, hi, lo))
    for _, key, rho, hi, lo in sorted(rows, reverse=True):
        print("%-28s %8.2f %10.4f %10.4f" % (key, rho, hi, lo))
    best = max(history, key=lambda r: r["f1"])
    print("\nbest trial F1 %.4f" % best["f1"])
    for k, v in sorted(best["config"].items()):
        print("  %-28s %s" % (k, v))


if __name__ == "__main__":
    main()
