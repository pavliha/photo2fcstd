import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photo2fcstd import view_rank

PHOTOS = os.path.expanduser("~/3DPrint/tools/data/printcad/PrintCAD/captured_img")
PICKED = "#2f8f4e"
DEFAULT = "#b0522f"


def photo(part, shot):
    hits = glob.glob(os.path.join(PHOTOS, "*", "%s_%d.jpg" % (part, shot + 1)))
    return hits[0] if hits else None


def out_of_fold(recs):
    import numpy as np
    from sklearn.model_selection import GroupKFold
    groups = json.load(open("data/part_groups.json"))
    g = np.array([groups.get(r["part"], -1) for r in recs])
    picks = {}
    for train_idx, test_idx in GroupKFold(n_splits=5).split(recs, groups=g):
        model = view_rank.train([recs[i] for i in train_idx])
        for i in test_idx:
            scores = model.predict(view_rank.matrix(recs[i]["shots"]))
            picks[recs[i]["part"]] = int(np.argmax(scores))
    return picks


def rows(recs, picks, want, key):
    ranked = sorted(
        ({"rec": r, "pick": picks[r["part"]]} for r in recs if r["part"] in picks),
        key=lambda e: key(e["rec"]["shots"], e["pick"]),
    )
    return [e for e in ranked if e["pick"] != 0][:want]


def draw(entries, path):
    fig, axes = plt.subplots(len(entries), 3, figsize=(9.5, 3.15 * len(entries)))
    fig.patch.set_facecolor("#faf8f5")
    for row, entry in zip(axes, entries):
        shots = entry["rec"]["shots"]
        for i, ax in enumerate(row):
            img = photo(entry["rec"]["part"], i)
            ax.imshow(plt.imread(img)) if img else ax.text(0.5, 0.5, "missing", ha="center")
            ax.set_xticks([]), ax.set_yticks([])
            chosen = i == entry["pick"]
            colour = PICKED if chosen else (DEFAULT if i == 0 else "#d8d2c8")
            for s in ax.spines.values():
                s.set_edgecolor(colour), s.set_linewidth(5 if chosen or i == 0 else 1)
            tag = "ranker picks" if chosen else ("old rule: photo 1" if i == 0 else "")
            ax.set_title("IoU %.2f   %s" % (shots[i]["iou"], tag), fontsize=9,
                         color=colour if tag else "#8a8177", pad=4)
        delta = shots[entry["pick"]]["iou"] - shots[0]["iou"]
        row[0].set_ylabel("part %s\n%+.2f" % (entry["rec"]["part"], delta), fontsize=9)
    fig.suptitle("Which photo we trace, and what it costs", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    return path


def main():
    recs = json.load(open("data/view_features.json"))
    gain = lambda shots, pick: shots[0]["iou"] - shots[pick]["iou"]
    loss = lambda shots, pick: shots[pick]["iou"] - shots[0]["iou"]
    picks = out_of_fold(recs)
    entries = rows(recs, picks, 4, gain) + rows(recs, picks, 2, loss)
    out = sys.argv[1] if len(sys.argv) > 1 else "tools/view_panel.png"
    print(draw(entries, out))


if __name__ == "__main__":
    main()
