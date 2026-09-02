"""How well the pipeline does at each level of part complexity - the curriculum ladder."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import bench, stats

TIERS = ((1, 4, "1-4 primitives"), (5, 8, "5-8"), (9, 16, "9-16"), (17, 10 ** 6, "17 or more"))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/trim_off"
    rows = []
    for part, row in bench.sketch_scores(run).items():
        value = row.get("primitive_f1")
        if not row.get("trustworthy") or not isinstance(value, dict) or not value.get("wanted"):
            continue
        rows.append({"part": part, "f1": float(value["f1"]), "wanted": int(value["wanted"]),
                     "drawn": int(value["drawn"]), "region": row.get("region_iou"),
                     "holes": max(int(row.get("loops_ideal") or 1) - 1, 0)})
    print("%s: %d parts with trustworthy truth\n" % (run, len(rows)))
    print("  %-16s %5s  %-22s %-16s %s" % ("tier", "n", "primitive F1", "region IoU", "drawn/wanted"))
    for lo, hi, label in TIERS:
        sel = [r for r in rows if lo <= r["wanted"] <= hi]
        if len(sel) < 5:
            continue
        m, a, b = stats.mean_ci([r["f1"] for r in sel])
        ri = np.mean([r["region"] for r in sel if r["region"] is not None])
        ratio = np.mean([r["drawn"] / max(r["wanted"], 1) for r in sel])
        print("  %-16s %5d  %.3f [%.3f, %.3f]   %.3f            %.2f" % (label, len(sel), m, a, b, ri, ratio))
    for label, keep in (("no holes", lambda r: r["holes"] == 0), ("with holes", lambda r: r["holes"] > 0)):
        sel = [r for r in rows if keep(r)]
        if len(sel) < 5:
            continue
        m, a, b = stats.mean_ci([r["f1"] for r in sel])
        print("  %-16s %5d  %.3f [%.3f, %.3f]" % (label, len(sel), m, a, b))


if __name__ == "__main__":
    main()
