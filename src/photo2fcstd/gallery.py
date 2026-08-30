import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Patch

from photo2fcstd.overlay import compare, draw
from photo2fcstd.trace import load, outline, segment_photo, upright_mask

from photo2fcstd.settings import data_dir

D = data_dir()


def draw_sketch(ax, spec):
    rv, ol = spec.get("revolve"), spec.get("outline")
    if rv:
        P = np.array(rv["profile"] + rv["profile"][:1])
        ax.plot(P[:, 0], P[:, 1], "-", lw=1.5, color="C0")
        ax.plot(-P[:, 0], P[:, 1], "-", color="C0", alpha=0.3)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        for h in rv["holes"]:
            ax.add_patch(Circle((h["cx"], -0.6 * rv["R"] + 0.3 * h["cy"]), h["r"], fill=False, lw=1.2, color="C3"))
        title = "revolve: half profile + %d holes" % len(rv["holes"])
    elif ol:
        n = {"circle": 0, "arc": 0, "line": 0}
        for l in ol["loops"]:
            if l["type"] == "circle":
                n["circle"] += 1
                ax.add_patch(Circle((l["cx"], l["cy"]), l["r"], fill=False, lw=1.5, color="C3"))
                continue
            for e in l["elements"]:
                n[e["type"]] += 1
                if e["type"] == "line":
                    ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.5, color="C0")
                else:
                    a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
                    a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
                    t1, t2 = (a0, a1) if e["ccw"] else (a1, a0)
                    ax.add_patch(Arc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"], theta1=t1, theta2=t2, lw=1.5, color="C2"))
        title = "sketch: %d circles, %d arcs, %d lines" % (n["circle"], n["arc"], n["line"])
    else:
        for name, v in spec["views"].items():
            z = np.array(v["z"])
            w = np.array([s["width"] for s in v["stations"]]) / 2
            xs, zs = [], []
            for i in range(len(w)):
                xs += [w[i], w[i]]
                zs += [z[i], z[i + 1]]
            off = 0 if name == "front" else max(w) * 2.5
            ax.plot(np.array(xs) + off, zs, "-", lw=1.5, color="C0" if name == "front" else "C1")
            ax.plot(-np.array(xs) + off, zs, "-", lw=1.5, color="C0" if name == "front" else "C1")
        title = "stations: %s" % ", ".join("%s %d" % (k, len(v["stations"])) for k, v in spec["views"].items())
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.08)
    ax.set_title(title, fontsize=8)


def main(run, out, per_mode=3):
    rows = [l.split() for l in open(os.path.join(run, "results.txt")) if len(l.split()) == 3 and l.split()[2] != "fail"]
    rows = [(k, m, float(v)) for k, m, v in rows]
    picked = []
    for mode in ("stations", "profile", "plan", "revolve"):
        r = sorted([x for x in rows if x[1] == mode], key=lambda x: x[2])
        if len(r) >= 3:
            picked += [r[-1], r[len(r) // 2], r[0]][:per_mode]
    fig, ax = plt.subplots(len(picked), 5, figsize=(17, 3.2 * len(picked)), squeeze=False)
    for i, (k, mode, iou) in enumerate(picked):
        spec = json.load(open(os.path.join(run, "out", k + ".spec.json")))
        src = (spec.get("revolve") or spec.get("outline") or spec["views"]["front"])["source"]
        c = compare(os.path.join(D, "stl_from_step", k + ".stl"), os.path.join(run, "out", k + ".stl"))
        ax[i, 0].imshow(load(src))
        ax[i, 0].set_title("%s  %s   3D IoU %.2f  missing %.0f%%  extra %.0f%%" % (k, mode, c["iou3d"], 100 * c["missing3d"], 100 * c["extra3d"]), fontsize=8, fontweight="bold", loc="left")
        mask, _ = upright_mask(segment_photo(src))
        poly, sh = outline(mask)
        ax[i, 1].imshow(mask, cmap="gray")
        ax[i, 1].plot(*np.vstack([poly, poly[:1]]).T, "r-", lw=1)
        for h in sh["holes"]:
            h = np.array(h)
            ax[i, 1].plot(*np.vstack([h, h[:1]]).T, "c-", lw=1)
        ax[i, 1].set_title("cutout + trace", fontsize=8)
        draw_sketch(ax[i, 2], spec)
        draw(ax[i, 3], c["views"]["face"], "face overlay")
        draw(ax[i, 4], c["views"]["side"], "side overlay")
    for a in ax.ravel():
        a.axis("off")
    v = np.array([x[2] for x in rows])
    fig.legend(handles=[Patch(color=(0.55, 0.55, 0.55), label="both"), Patch(color=(0.85, 0.15, 0.15), label="truth only (missing)"), Patch(color=(0.15, 0.35, 0.85), label="model only (extra)")], loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("%s — %d parts, mean IoU %.3f, median %.3f — rows: best / median / worst per mode" % (os.path.basename(run.rstrip("/")), len(v), v.mean(), np.median(v)), fontsize=11)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    plt.savefig(out, dpi=65)
    print("wrote", out, "rows", [(k, m, round(i, 2)) for k, m, i in picked])


def run():
    main(*sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "gallery.png"))
