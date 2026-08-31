"""Draw the ideal sketch beside what each path actually produces, on the same parts."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from axis_data import IDEAL, PERM, carve_part  # noqa: E402
from photo2fcstd import analysis, bench, carve as C, sketch_score as SS, spec as spec_mod  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))


def paint(ax, rings, color, title, iou=None):
    for r in rings:
        r = SS.normalise(np.asarray(r, float)) if False else np.asarray(r, float)
        ax.plot(*np.vstack([r, r[:1]]).T, "-", lw=1.6, color=color)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if iou is None else "%s   IoU %.2f" % (title, iou), fontsize=9,
                 color=color, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(2.0)


def carve_spec(carved, part, axis):
    cc = dict(carved)
    cc["points_mm"] = carved["points_mm"][:, PERM[axis]]
    return C.spec_from_carve(cc, name=part, axis=2)


def main(n=5):
    data = json.load(open(os.path.join(ROOT, "data", "axis_holdout.json")))
    have = set(bench.with_photos([r["part"] for r in data])[0])
    strata = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "data", "axis_strata.json")))}
    rows = [r for r in data if r["part"] in have
            and not strata.get(r["part"], {"circle": True})["circle"]
            and strata[r["part"]]["px"] >= 12]
    scored = []
    for r in rows:
        learned = int(np.argmax(MODEL.predict_proba(np.array(r["x"], float))[:, 1]))
        thin = int(np.argmax([f[10] for f in r["x"]]))
        if learned != thin:
            scored.append((r["iou"][learned] - r["iou"][thin], r, learned, thin))
    picked = sorted(scored, key=lambda t: -t[0])[:n]

    fig, ax = plt.subplots(len(picked), 4, figsize=(13, 3.3 * len(picked)), squeeze=False)
    for i, (_, r, learned, thin) in enumerate(picked):
        part = r["part"]
        carved = carve_part(part)
        paint(ax[i, 0], SS.ideal_rings(IDEAL[part]), "#111", "the real sketch (STEP)")
        try:
            views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
            doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
            paint(ax[i, 1], SS.spec_rings(doc), "#8a6d1f", "from one photo",
                  SS.score_one(doc, IDEAL[part])["region_iou"])
        except Exception:
            ax[i, 1].set_title("from one photo — failed", fontsize=9)
            ax[i, 1].axis("off")
        paint(ax[i, 2], SS.spec_rings(carve_spec(carved, part, thin)), "#b32d2e",
              "carved, thinnest-extent axis", r["iou"][thin])
        paint(ax[i, 3], SS.spec_rings(carve_spec(carved, part, learned)), "#1a7f37",
              "carved, learned axis", r["iou"][learned])
        ax[i, 0].set_ylabel(part, fontsize=11, fontweight="bold")
    fig.suptitle("Parts whose face is a real outline, not a circle — so region IoU cannot flatter the result",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(ROOT, "docs", "figures", "axis_sketches.png")
    plt.savefig(out, dpi=85)
    print("wrote", out)


if __name__ == "__main__":
    main()
