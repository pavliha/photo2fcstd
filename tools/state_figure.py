"""The state of photo2fcstd after the week of measurement, in one figure."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD, BAD, DIM, INK = "#0e6f5c", "#b3382c", "#9a9a94", "#1a1a1a"


def attempts(ax):
    rows = [("mode selection (selector)", 0.080, GOOD), ("view choice (selector)", 0.038, GOOD),
            ("chain_arcs (geometry)", 0.016, GOOD),
            ("BOP-trained, real labels (16th)", -0.003, DIM),
            ("seqnet + gates", 0.001, DIM), ("archive real labels", 0.000, DIM),
            ("section-trained model", 0.000, DIM),
            ("composition prior", -0.010, BAD), ("tangent merge", -0.005, BAD),
            ("DP decomposer (best sigma)", -0.032, BAD),
            ("seqnet + wobble", -0.034, BAD), ("seqnet raw", -0.067, BAD)]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    ax.axvline(0, color=INK, lw=0.8)
    for yi, (name, v, c) in zip(y, rows):
        ax.text(-0.003 if v >= 0 else 0.003, yi, name + "  ", ha="right" if v >= 0 else "left",
                va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.085, 0.095)
    ax.set_xlabel("end-to-end delta on photographs", fontsize=8)
    ax.set_title("selection and geometry win; sixteen replacements end at or below zero\n"
                 "grey = statistically zero, the best any replacement achieved", fontsize=9)


def dp_matrix(ax):
    cells = [("perfect\nideals", 0.064, "saturation\nbroken"),
             ("photographs", -0.066, "wobble tiled\nwith arcs"),
             ("photographs,\nmeasured sigma", -0.032, "halved,\nnot closed"),
             ("carve\nsections", -0.057, "right count,\nwrong places")]
    x = np.arange(len(cells))
    vals = [c[1] for c in cells]
    ax.bar(x, vals, color=[GOOD if v > 0 else BAD for v in vals], width=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    for xi, (name, v, note) in zip(x, cells):
        ax.text(xi, v + (0.005 if v > 0 else -0.005), note, ha="center",
                va="bottom" if v > 0 else "top", fontsize=7)
        ax.text(xi, 0.093 if v < 0 else -0.05, name, ha="center", va="top", fontsize=8)
    ax.set_xticks([]); ax.set_ylim(-0.115, 0.1)
    ax.set_title("the optimal decomposer: breaks the clean-input ceiling,\n"
                 "dies on every real input - greed is the accidental regulariser", fontsize=9)
    ax.set_ylabel("DP minus greedy, primitive F1", fontsize=8)


def tiers(ax):
    labels = ["1-4\nprimitives", "5-8", "9-16", "17+"]
    f1 = [0.651, 0.740, 0.696, 0.391]
    ax.bar(np.arange(4), f1, color=[GOOD, GOOD, GOOD, BAD], width=0.6)
    ax.axhline(0.918, color=INK, ls="--", lw=1)
    ax.text(3.45, 0.928, "usable ceiling 0.918", fontsize=7, ha="right")
    ax.axhline(0.637, color=DIM, ls=":", lw=1)
    ax.text(-0.4, 0.60, "overall 0.637", fontsize=7, color=DIM)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.015, "%.2f" % v, ha="center", fontsize=8)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_yticks([])
    ax.set_title("primitive F1 by tier, frozen test set, current code\n"
                 "17+ is the tracer's own ceiling; most of the rest is capture", fontsize=9)


def rig(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("what the sixteen-photo session buys - every number pre-measured", fontsize=9)
    rows = [("depth", "a 10x band nobody trusts", "measured to 1-5%", GOOD),
            ("solid IoU", "0.43", "~0.79", GOOD),
            ("twelve rim-shot discs", "F1 0.000, unreachable", "scoreable", GOOD),
            ("tilt reshoot check", "gated off, untrainable", "trained for good", GOOD),
            ("the drawing", "0.564", "0.529 - parity", DIM)]
    for i, (name, before, after, col) in enumerate(rows):
        y = 8.6 - i * 1.75
        ax.text(0.2, y, name, fontsize=10, fontweight="bold", va="center")
        ax.text(4.4, y, before, fontsize=9, va="center", ha="center", color=DIM)
        ax.annotate("", (6.4, y), (5.9, y), arrowprops=dict(arrowstyle="->", color=INK))
        ax.text(8.2, y, after, fontsize=9, va="center", ha="center", color=col,
                fontweight="bold" if col == GOOD else "normal")


def main(out="docs/figures/state.png"):
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
    attempts(ax[0, 0])
    dp_matrix(ax[0, 1])
    tiers(ax[1, 0])
    rig(ax[1, 1])
    fig.suptitle("photo2fcstd - the week measured out; one action left: docs/shoot.md", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(ROOT, out), dpi=85, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
