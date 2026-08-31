"""Show the photos the rules chose against the ones the model chose, and what each drew."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, modes, sketch_score as SS, spec as spec_mod  # noqa: E402
from photo2fcstd.trace import load  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
A = json.load(open(os.path.join(ROOT, "data", "ab_view_shipped.json")))
B = json.load(open(os.path.join(ROOT, "data", "ab_view_view_model.json")))


def rings(ax, rs, color, title):
    for r in rs:
        r = np.asarray(r, float)
        ax.plot(*np.vstack([r, r[:1]]).T, "-", lw=1.7, color=color)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.8)


def spec_for(part, views, use_model):
    modes.USE_VIEW_MODEL = use_model
    return spec_mod.assemble(views, name=part, log=lambda *a: None)


def main(n=5):
    gained = [(B[k]["iou"] - A[k]["iou"], k) for k in A
              if "iou" in A.get(k, {}) and "iou" in B.get(k, {})
              and A[k]["source"] != B[k]["source"] and 1 - A[k]["trivial"] >= 0.15]
    picks = [k for _, k in sorted(gained, reverse=True)[:n]]
    fig, ax = plt.subplots(len(picks), 5, figsize=(15, 3.1 * len(picks)), squeeze=False)
    for i, part in enumerate(picks):
        paths = bench.photos_of(part)[:3]
        views = [analysis.view(p) for p in paths]
        by_name = {os.path.basename(p): p for p in paths}
        for j, (arm, d, color) in enumerate((("rules", A, "#b32d2e"), ("model", B, "#1a7f37"))):
            src = by_name.get(d[part]["source"])
            ax[i, 2 * j].imshow(load(src) if src else np.zeros((10, 10)))
            ax[i, 2 * j].set_xticks([]); ax[i, 2 * j].set_yticks([])
            ax[i, 2 * j].set_title("%s chose %s" % (arm, d[part]["source"]), fontsize=9, color=color)
            doc = spec_for(part, views, arm == "model")
            rings(ax[i, 2 * j + 1], SS.spec_rings(doc), color,
                  "%s  IoU %.2f" % (d[part]["mode"], d[part]["iou"]))
        rings(ax[i, 4], SS.ideal_rings(IDEAL[part]), "#111", "the real sketch")
        ax[i, 0].set_ylabel(part, fontsize=10, fontweight="bold")
    modes.USE_VIEW_MODEL = True
    fig.suptitle("The rules drew from the wrong photo. Choosing the view is worth +0.038 sketch IoU.",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(ROOT, "docs", "figures", "view_choice.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
