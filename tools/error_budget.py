"""Attribute each part's shortfall to a cause we can act on."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import bench, stats

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/final"
IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))
LABELS = {r["part"]: r["iou_per_mode"] for r in json.load(open("data/mode_labels_full.json"))
          if r.get("iou_per_mode")}
TRUE_RATIO = {r["part"]: r["ratio"] for r in json.load(open("data/depth_rows.json"))}


def solids(run):
    out = {}
    for line in open(os.path.join(run, "results.txt")):
        f = line.split()
        if len(f) == 3 and f[2] != "fail":
            out[f[0]] = (f[1], float(f[2]))
    return out


def spec_of(run, part):
    path = os.path.join(run, "out", part + ".spec.json")
    return json.load(open(path)) if os.path.exists(path) else None


def hole_counts(part, doc):
    rec = IDEAL.get(part)
    want = len(rec["loops"]) - 1 if rec and rec.get("loops") else None
    ol = (doc or {}).get("outline") or {}
    got = max(len(ol.get("loops", [])) - 1, 0) if ol else None
    return got, want


def depth_error(part, doc):
    truth = TRUE_RATIO.get(part)
    ol = (doc or {}).get("outline") or {}
    if truth is None or not ol.get("depth_px"):
        return None
    views = doc.get("views") or {}
    src = os.path.basename(ol.get("source") or "")
    length = None
    if isinstance(views, dict):
        for v in views.values():
            if isinstance(v, dict) and os.path.basename(v.get("source", "")) == src:
                length = v.get("length_px")
    if not length:
        return None
    return abs(np.log(max(float(ol["depth_px"]) / float(length), 1e-6) / truth))


def main():
    got = solids(RUN)
    sketches = bench.sketch_scores(RUN)
    rows = []
    for part, (mode, iou) in got.items():
        doc = spec_of(RUN, part)
        per_mode = LABELS.get(part, {})
        best_mode = max(per_mode, key=per_mode.get) if per_mode else None
        drawn, wanted = hole_counts(part, doc)
        rows.append({
            "part": part, "iou": iou, "mode": mode,
            "mode_loss": (per_mode[best_mode] - per_mode.get(mode, 0.0)) if best_mode else None,
            "depth_err": depth_error(part, doc),
            "holes_missed": (wanted - drawn) if (drawn is not None and wanted is not None) else None,
            "sketch": (sketches.get(part) or {}).get("region_iou"),
        })
    worst = sorted(rows, key=lambda r: r["iou"])[:len(rows) // 4]
    print("run %s: %d parts scored, worst quartile = %d parts (mean IoU %.3f)"
          % (RUN, len(rows), len(worst), np.mean([r["iou"] for r in worst])))
    causes = {
        "mode choice costs >0.10": lambda r: (r["mode_loss"] or 0) > 0.10,
        "depth off by >1.5x": lambda r: (r["depth_err"] or 0) > np.log(1.5),
        "misses >=2 holes": lambda r: (r["holes_missed"] or 0) >= 2,
        "sketch region IoU <0.4": lambda r: r["sketch"] is not None and r["sketch"] < 0.4,
    }
    for label, hit in causes.items():
        share = 100 * np.mean([hit(r) for r in worst])
        base = 100 * np.mean([hit(r) for r in rows])
        print("  %-26s %4.0f%% of the worst quartile, %4.0f%% overall" % (label, share, base))
    none_of = [r for r in worst if not any(hit(r) for hit in causes.values())]
    print("  %-26s %4.0f%% of the worst quartile" % ("no cause identified", 100 * len(none_of) / len(worst)))
    json.dump(rows, open("data/error_budget.json", "w"))


if __name__ == "__main__":
    main()
