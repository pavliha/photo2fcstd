"""How chain_arcs works, in pictures: the turn signature, and real parts before and after."""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc as MArc, Circle, Ellipse, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

INK, GOOD, BAD, DIM = "#1a1a1a", "#0e6f5c", "#b3382c", "#9a9a94"


def draw_spec(ax, spec, arc_color=GOOD):
    ol = spec.get("outline") or {}
    for l in ol.get("loops", []):
        if l["type"] == "circle":
            ax.add_patch(Circle((l["cx"], l["cy"]), l["r"], fill=False, lw=1.6, color=arc_color))
            continue
        if l["type"] == "ellipse":
            ax.add_patch(Ellipse((l["cx"], l["cy"]), 2 * l["a"], 2 * l["b"],
                                 angle=np.degrees(l["theta"]), fill=False, lw=1.6, color=arc_color))
            continue
        for e in l["elements"]:
            if e["type"] == "line":
                ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.5,
                        color=INK, solid_capstyle="round")
            elif e["type"] == "arc":
                a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
                a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
                t1, t2 = (a0, a1) if e.get("ccw") else (a1, a0)
                ax.add_patch(MArc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"], theta1=t1, theta2=t2,
                                  lw=2.0, color=arc_color))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def signature(ax, kind):
    if kind == "chain":
        a = np.radians(np.linspace(210, 330, 5))
        pts = np.c_[np.cos(a), np.sin(a)] * 100
        ok = True
        title = "an arc split into pieces:\nevery turn small, same way"
    elif kind == "corner":
        pts = np.array([[-90, 40], [0, -40], [90, 40]], float)
        ok = False
        title = "two edges at a corner:\none large turn"
    else:
        pts = np.array([[-90, 0], [-45, 22], [0, 0], [45, 22], [90, 0]], float)
        ok = False
        title = "a zigzag:\nturns alternate sign"
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=INK, lw=1.8, ms=3.5)
    for i in range(1, len(pts) - 1):
        v0, v1 = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        turn = np.degrees(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], float(v0 @ v1)))
        below = pts[i][1] < pts[:, 1].mean() if kind == "zigzag" else pts[i][1] >= 0
        ax.annotate("%+.0f°" % turn, pts[i], textcoords="offset points",
                    xytext=(0, -18 if below else 12),
                    ha="center", fontsize=8, color=GOOD if ok else BAD)
    ax.text(pts[:, 0].mean(), pts[:, 1].min() - 45, "→ one arc" if ok else "→ stays lines",
            ha="center", va="top", fontsize=10, fontweight="bold", color=GOOD if ok else BAD)
    ax.set_title(title, fontsize=9, pad=14)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(pts[:, 0].min() - 30, pts[:, 0].max() + 30)
    ax.set_ylim(pts[:, 1].min() - 75, pts[:, 1].max() + 40)
    for s in ax.spines.values():
        s.set_visible(False)


def main(out="docs/figures/chain.png"):
    from photo2fcstd.trace import load
    parts = ["00460", "00330", "00709", "00160"]
    by = json.load(open(os.path.join(ROOT, "data", "ab_chain.json")))
    fig = plt.figure(figsize=(15, 3.2 + 3.0 * len(parts)))
    gs = fig.add_gridspec(1 + len(parts), 3, height_ratios=[1.05] + [1] * len(parts),
                          hspace=0.3, wspace=0.12)
    for j, kind in enumerate(("chain", "corner", "zigzag")):
        signature(fig.add_subplot(gs[0, j]), kind)
    for i, part in enumerate(parts, start=1):
        off = json.load(open(os.path.join(ROOT, "runs", "chain_off", "out", part + ".spec.json")))
        on = json.load(open(os.path.join(ROOT, "runs", "chain_on", "out", part + ".spec.json")))
        src = (off.get("outline") or {}).get("source")
        a, b = by["off"][part], by["on"][part]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(load(src)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("%s" % part, fontsize=9, loc="left")
        ax = fig.add_subplot(gs[i, 1])
        draw_spec(ax, off, arc_color=GOOD)
        ax.set_title("before: %d curved of %d wanted   F1 %.2f" % (a["curve_mine"], a["curve_ideal"], a["f1"]),
                     fontsize=9, color=DIM)
        ax = fig.add_subplot(gs[i, 2])
        draw_spec(ax, on, arc_color=GOOD)
        ax.set_title("after: %d curved   F1 %.2f" % (b["curve_mine"], b["f1"]), fontsize=9, color=GOOD)
    fig.suptitle("chain_arcs - three or more line pieces that all bend the same way become one arc\n"
                 "the merged run must still pass every shipped gate; photographs n=351: primitive F1 "
                 "+0.012 [+0.006, +0.020], builds identical", fontsize=11)
    fig.savefig(os.path.join(ROOT, out), dpi=80, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
