import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
from axis_data import carve_part  # noqa: E402
from photo2fcstd import carve as C  # noqa: E402

rows = json.load(open(os.path.join(ROOT, "data", "axis_holdout.json")))
model = joblib.load(os.path.join(ROOT, "data", "axis_model.joblib"))
best = np.array([r["best"] for r in rows])
thin = np.array([int(np.argmax([f[10] for f in r["x"]])) for r in rows])
pred = np.array([int(np.argmax(model.predict_proba(np.array(r["x"], float))[:, 1])) for r in rows])
iou = np.array([r["iou"] for r in rows], float)

fixed = [i for i in range(len(rows)) if pred[i] == best[i] and thin[i] != best[i]]
fixed = sorted(fixed, key=lambda i: iou[i][thin[i]] - iou[i][pred[i]])[:5]

fig, ax = plt.subplots(len(fixed), 3, figsize=(9.5, 3.1 * len(fixed)), squeeze=False)
for r, i in enumerate(fixed):
    row = rows[i]
    carved = carve_part(row["part"])
    p = model.predict_proba(np.array(row["x"], float))[:, 1]
    for a in (0, 1, 2):
        grid = C.occupancy(carved, axis=a)
        picked = "learned" if a == pred[i] else ("thinnest" if a == thin[i] else "")
        color = "#1a7f37" if a == pred[i] else ("#b32d2e" if a == thin[i] else "#999")
        ax[r, a].imshow(grid, cmap="gray_r", interpolation="nearest")
        ax[r, a].set_title("%s   IoU %.2f   score %.2f%s" % ("XYZ"[a], row["iou"][a], p[a],
                                                             "\n" + picked if picked else "\n"),
                           fontsize=9, color=color, fontweight="bold" if picked else "normal")
        for s in ax[r, a].spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2.5 if picked else 0.5)
        ax[r, a].set_xticks([])
        ax[r, a].set_yticks([])
    ax[r, 0].set_ylabel(row["part"], fontsize=10, fontweight="bold")

fig.suptitle("Which way to look at a carved part: green is what the classifier picked,\n"
             "red is the thinnest-extent rule it replaced", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(ROOT, "docs", "figures", "axis_model.png")
plt.savefig(out, dpi=85)
print("wrote", out)
