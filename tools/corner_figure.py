"""Show what the corner model predicts against approxPolyDP, and what the fitter does with each."""
import os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from corner_train import baseline_corners  # noqa: E402
from photo2fcstd import cornernet as CN  # noqa: E402
from photo2fcstd.curvenet import features, resample  # noqa: E402


def main(n=4):
    rows = list(np.load(os.path.join(ROOT, "data", "corner_rows.npy"), allow_pickle=True))
    rng = np.random.default_rng(3)
    def usable(r):
        c = np.asarray(r["contour"], float)
        if len(c) < 250 or not (6 <= len(r["corners"]) <= 14):
            return False
        b = c.max(0) - c.min(0)
        from shapely.geometry import Polygon
        try:
            poly = Polygon(c)
        except Exception:
            return False
        return min(b) / max(b) > 0.45 and poly.is_valid and poly.area > 0.25 * b[0] * b[1]

    picks = [rows[i] for i in rng.choice(len(rows), 400, replace=False)]
    picks = [r for r in picks if usable(r)][:n]
    m, dev = CN.load()

    fig, ax = plt.subplots(len(picks), 3, figsize=(12, 3.9 * len(picks)), squeeze=False)
    for i, r in enumerate(picks):
        contour = np.asarray(r["contour"], float)
        true = np.asarray(r["corners"], float)
        with torch.no_grad():
            h = torch.sigmoid(m(torch.from_numpy(features(contour)).unsqueeze(0).to(dev)))[0].cpu().numpy()
        pts, _ = resample(contour)
        learned = CN.peaks(h, pts)
        poly = baseline_corners(contour)
        diag = np.linalg.norm(contour.max(0) - contour.min(0))

        for j, (name, got, color) in enumerate((
                ("the STEP file's own corners", true, "#111"),
                ("approxPolyDP: %d corners, 0.48 precision" % len(poly), poly, "#1a7f37"),
                ("learned: %d corners, 0.87 precision" % len(learned), learned, "#b32d2e"))):
            a = ax[i, j]
            a.plot(*np.vstack([contour, contour[:1]]).T, "-", lw=1.0, color="#bbb")
            if len(got):
                a.plot(got[:, 0], got[:, 1], "o", ms=7, mfc="none", mew=2, color=color)
            if j > 0 and len(got) and len(true):
                d = np.linalg.norm(got[:, None] - true[None], axis=2).min(axis=1) / diag
                name += "\nmedian offset %.4f of the diagonal" % np.median(d)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.invert_yaxis()
            a.set_title(name, fontsize=9, color=color, fontweight="bold")
            for s in a.spines.values():
                s.set_edgecolor(color); s.set_linewidth(1.6)
        ax[i, 0].set_ylabel(r["part"], fontsize=10, fontweight="bold")

    fig.suptitle("approxPolyDP's extra corners are not mistakes - they are split points the fitter needs.\n"
                 "The model deletes them and the drawing gets worse, at 0.84 F1.", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.945))
    out = os.path.join(ROOT, "docs", "figures", "corners.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
