"""What the thing actually does: a photograph in, a sketch out, beside the sketch the part has."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402
from photo2fcstd.trace import load  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
INK, DRAW = "#111", "#1a5fb4"


def draw_spec(ax, doc):
    """The sketch as primitives - arcs as arcs, circles as circles - not as a polyline."""
    ol = doc.get("outline") or {}
    n = {"line": 0, "arc": 0, "circle": 0}
    for lp in ol.get("loops", []):
        if lp["type"] == "circle":
            n["circle"] += 1
            ax.add_patch(Circle((lp["cx"], lp["cy"]), lp["r"], fill=False, lw=1.9, color=DRAW))
            continue
        for e in lp["elements"]:
            n[e["type"]] += 1
            if e["type"] == "line":
                ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.9, color=DRAW)
            else:
                a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
                a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
                t1, t2 = (a0, a1) if e["ccw"] else (a1, a0)
                ax.add_patch(Arc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"],
                                 theta1=t1, theta2=t2, lw=1.9, color=DRAW))
    return "%d lines, %d arcs, %d circles" % (n["line"], n["arc"], n["circle"])


def draw_rings(ax, rings, colour):
    for r in rings:
        r = np.asarray(r, float)
        ax.plot(*np.vstack([r, r[:1]]).T, "-", lw=1.7, color=colour)


def frame(ax, title, colour=INK):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color=colour)
    for s in ax.spines.values():
        s.set_edgecolor("#ccc"); s.set_linewidth(1.0)


def main():
    scored = json.load(open(os.path.join(ROOT, "data", "ab_view_view_model.json")))
    rows = [(k, v["iou"]) for k, v in scored.items()
            if "iou" in v and 1 - v["trivial"] >= 0.15]
    rows.sort(key=lambda kv: -kv[1])
    picks = [("best", rows[0]), ("upper quartile", rows[len(rows) // 4]),
             ("median", rows[len(rows) // 2]), ("lower quartile", rows[3 * len(rows) // 4]),
             ("worst", rows[-1])]
    fig, ax = plt.subplots(len(picks), 4, figsize=(13.5, 3.35 * len(picks)), squeeze=False)
    for i, (label, (part, iou)) in enumerate(picks):
        photos = bench.photos_of(part)[:3]
        views = [analysis.view(p) for p in photos]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        used = (doc.get("outline") or doc.get("revolve") or {}).get("source", "")
        chosen = next((p for p in photos if os.path.basename(p) == os.path.basename(used)), photos[0])

        ax[i, 0].imshow(load(chosen))
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        ax[i, 0].set_title("the photograph it chose", fontsize=9)
        ax[i, 0].set_ylabel("%s\n%s  IoU %.2f" % (part, label, iou), fontsize=9.5, fontweight="bold")

        mask, _ = analysis.upright_mask(analysis.segment_photo(chosen))
        ax[i, 1].imshow(mask, cmap="gray_r")
        ax[i, 1].set_xticks([]); ax[i, 1].set_yticks([])
        ax[i, 1].set_title("silhouette", fontsize=9)

        counts = draw_spec(ax[i, 2], doc)
        frame(ax[i, 2], "the sketch it drew\n" + counts, DRAW)
        ax[i, 2].autoscale_view()
        ax[i, 2].relim(); ax[i, 2].autoscale()

        draw_rings(ax[i, 3], SS.ideal_rings(IDEAL[part]), INK)
        frame(ax[i, 3], "the sketch the part has")

    fig.suptitle("photo2fcstd end to end: one photograph in, a parametric sketch out\n"
                 "five parts spanning the whole range, not a selection", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(ROOT, "docs", "figures", "photo_to_sketch.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
