"""Follow one outline through the tracer and see where its corners disappear."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, modes, sketch_score as SS, trace  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
LOST, KEPT, INK = "#b32d2e", "#1a7f37", "#111"


def plot_elements(ax, els, colour, title, sub=""):
    for e in els:
        ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=2.0, color=colour)
    v = np.array([e["p0"] for e in els], float)
    ax.plot(v[:, 0], v[:, 1], "o", ms=6, mfc="white", mew=1.8, color=colour)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
    ax.set_title("%s\n%s" % (title, sub), fontsize=9.5, color=colour, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(1.8)


def main(part="00911", view_index=1):
    p = bench.photos_of(part)[view_index]
    v = analysis.view(p)
    raw = np.asarray(v["shape"]["raw"], float)
    cx, cy = raw.mean(0)
    outer = np.asarray(modes.centred_loops(v, cx, cy)[0], float)
    L = v["length_px"]

    fig, ax = plt.subplots(1, 5, figsize=(19, 4.4))

    ax[0].plot(outer[:, 0], outer[:, 1], "-", lw=1.0, color="#777")
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].invert_yaxis()
    ax[0].set_title("the traced contour\n%d points, every notch present" % len(outer),
                    fontsize=9.5, color=INK, fontweight="bold")
    for s in ax[0].spines.values():
        s.set_edgecolor(INK); s.set_linewidth(1.8)

    runs = trace.corner_runs(outer)
    corners = np.array([r[0] for r in runs], float)
    ax[1].plot(outer[:, 0], outer[:, 1], "-", lw=1.0, color="#ccc")
    ax[1].plot(corners[:, 0], corners[:, 1], "o", ms=7, mfc="white", mew=2, color=LOST)
    ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].invert_yaxis()
    ax[1].set_title("approxPolyDP corners\n%d found, the part has 12" % len(runs),
                    fontsize=9.5, color=LOST, fontweight="bold")
    for s in ax[1].spines.values():
        s.set_edgecolor(LOST); s.set_linewidth(1.8)

    els = trace.elements(outer, L)
    plot_elements(ax[2], els, LOST, "after fitting", "%d elements" % len(els))
    reg = trace.regularise_lines([dict(e) for e in els], L)
    plot_elements(ax[3], reg, LOST, "after regularising", "%d elements" % len(reg))
    rect = trace.rectangularise([dict(e) for e in reg])
    plot_elements(ax[4], rect, LOST, "after rectangularise", "%d elements" % len(rect))

    fig.suptitle("Where part %s's notches go: the contour has them, approxPolyDP does not.\n"
                 "Twelve corners become eight before any regularisation runs; the last pass takes the remaining four."
                 % part, fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(ROOT, "docs", "figures", "where_notches_go.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)
    print("contour %d pts -> approxPolyDP %d corners -> fitted %d -> regularised %d -> rectangularised %d; truth 12"
          % (len(outer), len(runs), len(els), len(reg), len(rect)))


if __name__ == "__main__":
    main(*sys.argv[1:])
