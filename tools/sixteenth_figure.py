"""Tonight's results: the label factory, the neutral sixteenth, the pixel null."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD, BAD, DIM, INK = "#0e6f5c", "#b3382c", "#9a9a94", "#1a1a1a"


def labelled_contours(fig, gs_row):
    d = np.load(os.path.join(ROOT, "data", "bop_seq.npz"), allow_pickle=True)
    X, seq, G = d["X"], d["seq"], d["group"]
    rng = np.random.default_rng(4)
    curvy = [i for i in range(len(X))
             if sum(int(seq[i][j]) == 1 for j in range(0, int((seq[i] >= 0).sum()), 2)) >= 2]
    picks = list(rng.choice(curvy, 3, replace=False)) + list(rng.choice(len(X), 2, replace=False))
    for c, i in enumerate(picks):
        ax = fig.add_subplot(gs_row[c])
        pts, s = X[i], seq[i]
        spans, j = [], 0
        while j + 1 < len(s) and s[j] >= 0:
            spans.append((int(s[j + 1]), int(s[j])))
            j += 2
        prev = 0
        for e, t in sorted(spans):
            seg = pts[prev:e + 1]
            ax.plot(seg[:, 0], seg[:, 1], "-", color=GOOD if t else INK, lw=1.8)
            ax.plot(pts[e, 0], pts[e, 1], ".", color=BAD, ms=5)
            prev = e
        ax.set_title(G[i], fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")


def sixteenth(ax):
    rows = [("seqnet raw\n(synthetic training)", -0.067),
            ("+ wobble", -0.034),
            ("+ acceptance gates", 0.001),
            ("16th: real BOP labels\n(deployment data)", -0.003)]
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.bar(x, vals, color=[BAD, BAD, DIM, DIM], width=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    for xi, (name, v) in zip(x, rows):
        ax.text(xi, -0.0745, name, ha="center", va="top", fontsize=8)
        ax.text(xi, v + (0.002 if v >= 0 else -0.002), "%+.3f" % v,
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks([]); ax.set_ylim(-0.095, 0.02); ax.set_yticks([])
    ax.set_title("the ratchet to neutrality: training on real deployment data\n"
                 "removed the harm - the first non-harmful replacement of sixteen", fontsize=9)


def factory(ax):
    rows = [("ITODD", 105, 1.7), ("IPD", 723, 1.4), ("XYZ-IBD", 26707, 1.7), ("T-LESS", 32587, 1.9)]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [np.log10(r[1]) for r in rows], color=GOOD, height=0.55)
    for yi, (name, n, px) in zip(y, rows):
        ax.text(0.05, yi, name, va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(np.log10(n) + 0.07, yi, "%,d usable  ·  %.1f px labels" % (n, px)
                if False else "{:,} usable   {:.1f} px labels".format(n, px),
                va="center", fontsize=8)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(0, 6.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("the label factory: 60,122 real photographs whose contours\n"
                 "label themselves through the pose (log scale)", fontsize=9)


def pixels(ax):
    ax.bar([0, 1], [0.4152, 0.4129], color=[DIM, BAD], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["contour only", "contour + light\n(normal profiles)"], fontsize=9)
    for xi, v in ((0, 0.4152), (1, 0.4129)):
        ax.text(xi, v + 0.002, "%.4f" % v, ha="center", fontsize=9)
    ax.set_ylim(0.38, 0.44); ax.set_yticks([])
    ax.set_title("the pixel screen: seven intensities along each point's normal\n"
                 "add nothing (-0.0022, two seeds) - light doesn't rescue the contour", fontsize=9)


def main(out="docs/figures/sixteenth.png"):
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.8, 1.1, 1.1], hspace=0.42)
    row = [gs[0, i] for i in (0, 1, 2, 3, 4)]
    labelled_contours(fig, row)
    ax_note = fig.add_subplot(gs[0, 5]); ax_note.axis("off")
    ax_note.text(0, 0.5, "real contours,\nself-labelled:\ngreen curved\nblack straight\nred breakpoints",
                 fontsize=9, va="center")
    sixteenth(fig.add_subplot(gs[1, :3]))
    factory(fig.add_subplot(gs[1, 3:]))
    pixels(fig.add_subplot(gs[2, 1:5]))
    fig.suptitle("the sixteenth attempt and its two screens - deployment data ends the harm; "
                 "neither corpus nor light ends the wall", fontsize=12)
    fig.savefig(os.path.join(ROOT, out), dpi=85, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
