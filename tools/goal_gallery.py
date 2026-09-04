"""The goal, visualised: photograph in, the sketch the STEP wants, the sketch the pipeline drew."""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from chain_figure import GOOD, BAD, DIM, draw_spec  # noqa: E402
import tracer_ceiling as TC  # noqa: E402

RUN = "runs/chain_shipped"
CURVED = {"arc", "circle", "ellipse", "bsplinecurve"}


def draw_ideal(ax, rec):
    for loop in rec["loops"]:
        for e in loop:
            p = np.asarray(e["xy"], float)
            ax.plot(p[:, 0], p[:, 1], "-", lw=1.6,
                    color=GOOD if e["type"] in CURVED else "#1a1a1a")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main(out="docs/figures/goal_gallery.png"):
    from photo2fcstd import bench
    from photo2fcstd.trace import load
    rows = []
    for part, r in bench.sketch_scores(RUN).items():
        v = r.get("primitive_f1")
        if not r.get("trustworthy") or not isinstance(v, dict) or not v.get("wanted"):
            continue
        if (1 - r.get("trivial", 0)) < 0.15:
            continue
        rows.append((float(v["f1"]), part, int(v["wanted"]), int(v["drawn"])))
    rows.sort(reverse=True)
    n = len(rows)
    picks = [rows[i] for i in (0, 1, n // 4, n // 2, n // 2 + 1, 3 * n // 4, n - 4, n - 1)]
    fig, ax = plt.subplots(len(picks), 3, figsize=(11.5, 3.1 * len(picks)))
    for r, (f1, part, wanted, drawn) in enumerate(picks):
        spec = json.load(open(os.path.join(ROOT, RUN, "out", part + ".spec.json")))
        src = (spec.get("outline") or spec.get("revolve") or {}).get("source") or bench.photos_of(part)[0]
        ax[r, 0].imshow(load(src))
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for s in ax[r, 0].spines.values():
            s.set_visible(False)
        ax[r, 0].set_title(part, fontsize=9, loc="left")
        draw_ideal(ax[r, 1], TC.IDEAL[part])
        ax[r, 1].set_title("expected: %d primitives (from the STEP)" % wanted, fontsize=9, color=DIM)
        rv = spec.get("revolve")
        if rv and not spec.get("outline"):
            from matplotlib.patches import Circle
            a2 = ax[r, 2]
            a2.add_patch(Circle((0, 0), rv["R"], fill=False, lw=1.6, color=GOOD))
            for h in rv.get("holes", []):
                if h.get("type") == "circle":
                    a2.add_patch(Circle((h["cx"], h["cy"]), h["r"], fill=False, lw=1.6, color=GOOD))
            a2.set_aspect("equal")
            a2.set_xlim(-1.15 * rv["R"], 1.15 * rv["R"]); a2.set_ylim(-1.15 * rv["R"], 1.15 * rv["R"])
            a2.set_xticks([]); a2.set_yticks([])
            for s in a2.spines.values():
                s.set_visible(False)
        else:
            draw_spec(ax[r, 2], spec)
        col = GOOD if f1 >= 0.8 else ("#8a6d00" if f1 >= 0.5 else BAD)
        ax[r, 2].set_title("actual: %d primitives   F1 %.2f" % (drawn, f1), fontsize=9, color=col)
    fig.suptitle("photograph -> the sketch the STEP wants -> the sketch the pipeline drew\n"
                 "rows sampled from best to worst on the frozen test set (F1 %.3f overall)"
                 % np.mean([f for f, *_ in rows]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(ROOT, out), dpi=80, bbox_inches="tight")
    print("wrote", out, " sampled:", [p for _, p, *_ in picks])


if __name__ == "__main__":
    main()
