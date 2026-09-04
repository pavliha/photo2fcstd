"""The simple parts, drawn beside the ideal they miss by over-drawing."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
from photo2fcstd.bench import sketch_scores
from photo2fcstd.gallery import draw_sketch
from photo2fcstd.sketch_score import ideal_rings

IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/review_check"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/simple_tier.png"


def spec_of(part):
    path = os.path.join(RUN, "out", part + ".spec.json")
    return json.load(open(path)) if os.path.exists(path) else None


def worst(rows, count):
    simple = [(p, r["primitive_f1"]) for p, r in rows.items()
              if isinstance(r.get("primitive_f1"), dict) and r["primitive_f1"].get("wanted")
              and r["primitive_f1"]["wanted"] <= 4 and r["primitive_f1"]["drawn"] > r["primitive_f1"]["wanted"]]
    return sorted(simple, key=lambda x: x[1]["drawn"] - x[1]["wanted"], reverse=True)[:count]


def run():
    rows = sketch_scores(RUN)
    picks = worst(rows, 6)
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for column, (part, f1) in enumerate(picks):
        top, bottom = axes[0][column], axes[1][column]
        for ring in ideal_rings(IDEAL.get(part) or {}):
            if len(ring):
                closed = np.vstack([ring, ring[:1]])
                top.plot(closed[:, 0], closed[:, 1], "-", lw=1.6, color="#2f8f4e")
        spec = spec_of(part)
        if spec:
            draw_sketch(bottom, spec)
        top.set_title("%s  ideal %d" % (part, f1["wanted"]), fontsize=9)
        bottom.set_title("drew %d   F1 %.2f" % (f1["drawn"], f1["f1"]), fontsize=9)
        for ax in (top, bottom):
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle("Simple parts (ideal 4 primitives or fewer) that over-draw", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    run()
