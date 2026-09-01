"""Show the two out-of-distribution gates: where they fire, and what they save."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2fcstd import bench, embed, sketch_score as SS, tless  # noqa: E402
from embed_shift import rgb_for, vectors  # noqa: E402

GREEN, RED, INK = "#1a7f37", "#b32d2e", "#111"


def main():
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p])]
    parts = bench.with_photos(parts)[0][150:230]
    pc = vectors([bench.photos_of(p)[0] for p in parts])
    tl_paths = []
    for obj in range(1, 31):
        try:
            for _, mp in tless.views_of(obj, every=200, limit=1):
                got = rgb_for(mp)
                if got:
                    tl_paths.append(got)
        except Exception:
            continue
    tl = vectors(tl_paths)
    ref = dict(np.load(os.path.join(ROOT, "data", "embed_centre.npz")))
    c = np.asarray(ref["centre"], float)
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    d_in, d_out = 1.0 - unit(pc) @ c, 1.0 - unit(tl) @ c
    gate = float(ref["p95"]) * 1.1

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    a = ax[0]
    bins = np.linspace(0, 1.0, 34)
    a.hist(d_in, bins=bins, color=GREEN, alpha=0.75, label="PrintCAD (n=%d)" % len(pc))
    a.hist(d_out, bins=bins, color=RED, alpha=0.75, label="T-LESS (n=%d)" % len(tl))
    a.axvline(gate, color=INK, ls="--", lw=1.8)
    a.text(gate + 0.012, a.get_ylim()[1] * 0.92, "gate", fontsize=9, fontweight="bold")
    a.set_xlabel("cosine distance to the photographs the head was fitted on")
    a.set_ylabel("images")
    a.set_title("The backbone puts the two datasets\nin places that do not overlap", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1]
    rows = [("T-LESS,\nungated", 1.117, RED), ("T-LESS,\ngated", 0.236, GREEN),
            ("best constant\non T-LESS", 0.314, "#888")]
    a.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], width=0.6)
    for i, r in enumerate(rows):
        a.text(i, r[1] + 0.03, "%.3f" % r[1], ha="center", fontsize=10, fontweight="bold")
    a.set_ylabel("median absolute log error (lower is better)")
    a.set_title("Refusing what it was never asked about\nis worth more than answering", fontsize=10)
    a.set_ylim(0, 1.3)

    a = ax[2]
    a.axis("off")
    a.text(0.0, 0.97, "Two out-of-distribution gates", fontsize=12, fontweight="bold", va="top")
    lines = [
        ("axis choice", "section_constancy", "no constant-section axis exists,\nso no base face does either",
         "36% to 56% on T-LESS,\nPrintCAD untouched"),
        ("depth, pixel path", "embedding_is_familiar", "the photograph sits where the head\nwas never fitted",
         "1.117 to 0.236 on T-LESS,\n98 of 100 PrintCAD admitted"),
    ]
    y = 0.86
    for name, fn, when, worth in lines:
        a.text(0.0, y, name, fontsize=11, fontweight="bold", va="top")
        a.text(0.0, y - 0.055, fn, fontsize=8.5, va="top", family="monospace", color="#555")
        a.text(0.0, y - 0.115, "fires when " + when, fontsize=8.5, va="top", color=RED)
        a.text(0.0, y - 0.225, worth, fontsize=8.5, va="top", color=GREEN, fontweight="bold")
        y -= 0.38
    a.text(0.0, 0.06, "Neither was findable in distribution:\non PrintCAD alone both signals are flat.",
           fontsize=9, va="top", style="italic")

    fig.suptitle("Both learned components that failed on unseen parts now refuse instead of guessing",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(ROOT, "docs", "figures", "ood_gates.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
