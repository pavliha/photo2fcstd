"""Show the axis model failing on parts it was never trained on, and the depth model beside it."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from photo2fcstd import axis_model, carve as C, tless  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))


def main(n=4):
    ref = json.load(open(os.path.join(ROOT, "data", "tless_sketch.json")))
    wrong = [r for r in ref if not r["axis_model_agreed"]][:n]
    fig = plt.figure(figsize=(13.5, 3.05 * (len(wrong) + 1)))
    gs = fig.add_gridspec(len(wrong) + 1, 4, height_ratios=[1] * len(wrong) + [1.15])

    for i, r in enumerate(wrong):
        carved, _ = tless.carve_object(r["obj"], every=90, voxel_mm=1.0)
        if carved is None:
            continue
        p = MODEL.predict_proba(axis_model.features(carved))[:, 1]
        pick = int(np.argmax(p))
        for a in (0, 1, 2):
            ax = fig.add_subplot(gs[i, a])
            ax.imshow(C.occupancy(carved, axis=a), cmap="gray_r", interpolation="nearest")
            role = "the model chose this" if a == pick else ("a sketch describes this" if a == r["axis"] else "")
            colour = "#b32d2e" if a == pick else ("#1a7f37" if a == r["axis"] else "#999")
            ax.set_title("%s   score %.2f\n%s" % ("XYZ"[a], p[a], role or " "),
                         fontsize=9, color=colour, fontweight="bold" if role else "normal")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_edgecolor(colour); s_.set_linewidth(2.2 if role else 0.5)
        ax = fig.add_subplot(gs[i, 3])
        ax.axis("off")
        ax.text(0.0, 0.85, "T-LESS object %02d" % r["obj"], fontsize=11, fontweight="bold", va="top")
        ax.text(0.0, 0.60, "section varies by %.2f\nalong its best axis" % r["variation"],
                fontsize=9, va="top", family="monospace", color="#555")
        ax.text(0.0, 0.28, "confident and wrong:\nit scores its own choice\n%.2f, the right one %.2f"
                % (p[pick], p[r["axis"]]), fontsize=9, va="top", color="#b32d2e")

    ax = fig.add_subplot(gs[len(wrong), :2])
    rows = json.load(open(os.path.join(ROOT, "data", "depth_ood.json")))
    t = np.array([x["true"] for x in rows]); q = np.array([x["pred"] for x in rows])
    ax.scatter(t, q, s=34, c="#b32d2e", zorder=3, label="T-LESS")
    lim = [0.05, 1.8]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.axhspan(0.055, 0.503, color="#1a7f37", alpha=0.10)
    ax.text(0.075, 0.30, "the range it was\ntrained on", fontsize=8, color="#1a7f37")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true depth / length"); ax.set_ylabel("predicted")
    ax.set_title("Depth model on T-LESS: it predicts the range it was shown", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[len(wrong), 2:])
    ax.axis("off")
    ax.text(0.0, 0.95, "Trained on PrintCAD, measured elsewhere", fontsize=12, fontweight="bold", va="top")
    lines = [("view choice", "0.56 agree", "survives its convention test", "#1a7f37"),
             ("axis choice", "89% correct", "29% on T-LESS, chance is 33%", "#b32d2e"),
             ("depth ratio", "beats a constant 2:1", "loses to one on T-LESS", "#b32d2e")]
    for j, (name, a, b, c) in enumerate(lines):
        y = 0.72 - j * 0.20
        ax.text(0.0, y, name, fontsize=10, fontweight="bold", va="top")
        ax.text(0.26, y, a, fontsize=9, va="top", family="monospace", color="#555")
        ax.text(0.58, y, b, fontsize=9, va="top", color=c, fontweight="bold")
    ax.text(0.0, 0.10, "split-by-part stops a model memorising a part.\nNothing here stopped one memorising a dataset.",
            fontsize=9.5, va="top", style="italic", color="#333")

    fig.suptitle("Two of three learned components did not survive a dataset they had never seen",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(ROOT, "docs", "figures", "generalisation.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
