"""The same failure, drawn as the sketches it actually produces."""
import json, os, sys

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from photo2fcstd import axis_model, carve as C, sketch_score as SS, tless  # noqa: E402
from tless_sketch import cross_section  # noqa: E402

MODEL = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
PERM = {0: [1, 2, 0], 1: [2, 0, 1], 2: [0, 1, 2]}


def draw(ax, rings, colour, title, lw=1.9):
    for r in rings:
        r = np.asarray(r, float)
        ax.plot(*np.vstack([r, r[:1]]).T, "-", lw=lw, color=colour)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, color=colour, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(2.0)


def spec_on(carved, axis, name):
    cc = dict(carved)
    cc["points_mm"] = carved["points_mm"][:, PERM[axis]]
    return C.spec_from_carve(cc, name=name, axis=2)


def main(n=4):
    ref = json.load(open(os.path.join(ROOT, "data", "tless_sketch.json")))
    wrong = [r for r in ref if not r["axis_model_agreed"]][:n]
    fig, ax = plt.subplots(len(wrong), 3, figsize=(11, 3.4 * len(wrong)), squeeze=False)
    for i, r in enumerate(wrong):
        carved, _ = tless.carve_object(r["obj"], every=90, voxel_mm=1.0)
        if carved is None:
            continue
        p = MODEL.predict_proba(axis_model.features(carved))[:, 1]
        pick = int(np.argmax(p))
        truth = cross_section(trimesh.load(tless.model_path(r["obj"])), r["axis"])
        record = {"loops": [[{"xy": np.asarray(x).tolist(), "type": "line"}] for x in truth],
                  "n_loops": len(truth), "prism": True, "counts": {}}
        s_bad = spec_on(carved, pick, "bad")
        s_good = spec_on(carved, r["axis"], "good")
        draw(ax[i, 0], SS.spec_rings(s_bad), "#b32d2e",
             "what ships: looking down %s\nIoU %.2f" % ("XYZ"[pick], SS.score_one(s_bad, record)["region_iou"]))
        draw(ax[i, 1], SS.spec_rings(s_good), "#1a7f37",
             "the right axis %s\nIoU %.2f" % ("XYZ"[r["axis"]], SS.score_one(s_good, record)["region_iou"]))
        draw(ax[i, 2], truth, "#111", "the part's real cross-section")
        ax[i, 0].set_ylabel("T-LESS %02d" % r["obj"], fontsize=10, fontweight="bold")
    fig.suptitle("The drawing the axis model produces on parts it was never trained on,\n"
                 "against the one the part actually has", fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.945))
    out = os.path.join(ROOT, "docs", "figures", "ood_sketches.png")
    plt.savefig(out, dpi=82)
    print("wrote", out)


if __name__ == "__main__":
    main()
