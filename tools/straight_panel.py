"""Before and after the straightening work, drawn from two runs' sketches."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
from photo2fcstd.gallery import draw_sketch
from photo2fcstd.trace import dominant_frame

POPULAR = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], float)


def doc_of(run, part):
    path = os.path.join(run, "out", part + ".spec.json")
    return json.load(open(path)) if os.path.exists(path) else None


def cleanliness(doc):
    loops = ((doc or {}).get("outline") or {}).get("loops") or []
    angles, frames = [], []
    for loop in loops:
        els = [e for e in loop.get("elements", []) if e["type"] == "line"]
        if len(els) >= 3:
            frames.append((sum(np.hypot(*np.subtract(e["p1"], e["p0"])) for e in els),
                           dominant_frame(np.array([e["p0"] for e in els], float))))
        for e in els:
            d = np.subtract(e["p1"], e["p0"])
            angles.append(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    if not angles:
        return None
    frame = max(frames)[1] if frames else 0.0
    gap = np.abs(np.array(angles)[:, None] - ((POPULAR + frame) % 180.0)[None, :])
    return float(np.mean(np.minimum(gap, 180.0 - gap).min(axis=1) <= 2.0))


def main():
    before_run, after_run = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "tools/straight_panel.png"
    parts = sorted(f.split(".")[0] for f in os.listdir(os.path.join(after_run, "out"))
                   if f.endswith(".spec.json"))
    scored = []
    for part in parts:
        a, b = doc_of(before_run, part), doc_of(after_run, part)
        ca, cb = cleanliness(a), cleanliness(b)
        if ca is None or cb is None or ca > 0.9:
            continue
        scored.append((cb - ca, part, ca, cb))
    scored.sort(reverse=True)
    picks = scored[:5]
    fig, ax = plt.subplots(len(picks), 2, figsize=(8.5, 4.0 * len(picks)), squeeze=False)
    fig.patch.set_facecolor("#faf8f5")
    for row, (delta, part, ca, cb) in zip(ax, picks):
        for cell, run, label, share in ((row[0], before_run, "before", ca), (row[1], after_run, "after", cb)):
            draw_sketch(cell, doc_of(run, part))
            cell.set_aspect("equal")
            cell.axis("off")
            cell.set_title("%s  %s\n%.0f%% of edges on a clean angle" % (part, label, 100 * share),
                           fontsize=9, color="#b0522f" if label == "before" else "#2f8f4e")
    fig.suptitle("Straightening the lines", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=105, facecolor=fig.get_facecolor())
    print(out, [(p, "%.2f->%.2f" % (a, b)) for _, p, a, b in picks])


if __name__ == "__main__":
    main()
