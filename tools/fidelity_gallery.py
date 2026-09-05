import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
GOOD, BAD, INK = "#0e6f5c", "#b3382c", "#1a1a1a"


def draw_ideal(ax, part):
    for loop in IDEAL[part]["loops"]:
        for e in loop:
            xy = np.asarray(e["xy"], float)
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.6, color=GOOD if e["type"] != "line" else INK)
    ax.set_aspect("equal"); ax.axis("off")


def gallery(rows, title, out, ncol=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chain_figure import draw_spec
    from photo2fcstd import bench, recognise
    from photo2fcstd.trace import load
    n = len(rows)
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.0 * n))
    for i, r in enumerate(rows):
        part = r["part"]
        photos = bench.photos_of(part)[:3]
        spec, _ = recognise.route(photos, name=part, rec={"single_extrusion": True, "face_photo_index": 0})
        src = (spec.get("outline") or {}).get("source") or photos[0]
        ax = axes[i, 0]; ax.imshow(load(src)); ax.axis("off")
        ax.set_title("%s  photo (chosen view)" % part, fontsize=8)
        ax = axes[i, 1]; draw_ideal(ax, part); ax.set_title("ground truth sketch (STEP)", fontsize=8)
        ax = axes[i, 2]
        try:
            if spec.get("outline"):
                draw_spec(ax, spec)
            else:
                ax.text(0.5, 0.5, "mode: %s (no outline sketch)" % spec.get("mode"), ha="center", fontsize=8)
        except Exception as ex:
            ax.text(0.5, 0.5, "draw failed: %s" % str(ex)[:40], ha="center", fontsize=7)
        ax.axis("off")
        ax.set_title("ours  F1 %.2f  IoU %s  %s" % (r["f1"], r["iou"], "REFUSED" if r["refused"] else ""),
                     fontsize=8, color=BAD if r["refused"] else INK)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in json.load(open(os.path.join(ROOT, "runs", "gate_bench.json"))) if "f1" in r and r.get("valid")]
    rows.sort(key=lambda r: r["f1"])
    n = len(rows)
    worst, mid, best = rows[:8], rows[n // 2 - 4:n // 2 + 4], rows[-8:]
    od = os.path.join(ROOT, "runs", "results")
    os.makedirs(od, exist_ok=True)
    outs = [gallery(worst, "WORST 8 of %d: where the drawing does not match the STEP truth" % n, os.path.join(od, "fidelity_worst.png")),
            gallery(mid, "MEDIAN 8: typical fidelity (F1 around %.2f)" % rows[n // 2]["f1"], os.path.join(od, "fidelity_median.png")),
            gallery(best, "BEST 8: where it matches", os.path.join(od, "fidelity_best.png"))]
    f1 = np.array([r["f1"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(f1, bins=20, color=GOOD, alpha=0.85)
    for v, lab in ((np.median(f1), "median %.2f" % np.median(f1)), (0.918, "usable ceiling 0.918")):
        ax.axvline(v, color=BAD if v < 0.9 else INK, ls="--"); ax.text(v + 0.01, ax.get_ylim()[1] * 0.9, lab, fontsize=8)
    ax.set_xlabel("primitive F1 vs STEP ground truth"); ax.set_ylabel("parts")
    ax.set_title("shape fidelity on %d PrintCAD parts (real photos): exact match %.0f%%, F1>=0.8 %.0f%%, F1<0.4 %.0f%%"
                 % (n, 100 * np.mean(f1 >= 0.999), 100 * np.mean(f1 >= 0.8), 100 * np.mean(f1 < 0.4)), fontsize=9)
    p = os.path.join(od, "fidelity_hist.png"); fig.savefig(p, dpi=95, bbox_inches="tight"); outs.append(p)
    print("WROTE", outs)


if __name__ == "__main__":
    main()
