"""What seqnet drew: it fixes exactly the saturation it was built for, and ruins the simple parts."""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from chain_figure import GOOD, BAD, draw_spec  # noqa: E402

WIN = [("00082", "curves 5 of 12 -> 12 of 12"), ("00389", "curves 4 of 16 -> 10"),
       ("00622", "curves 1 of 12 -> 11")]
LOSS = [("00063", "a rectangle drawn as 33 arcs"), ("00009", "29 arcs, ideal has none"),
        ("00740", "19 arcs, ideal has none")]


def main(out="docs/figures/seqnet.png"):
    from photo2fcstd.trace import load
    by = json.load(open(os.path.join(ROOT, "data", "ab_seq.json")))
    rows = WIN + LOSS
    fig, ax = plt.subplots(len(rows), 3, figsize=(12.5, 3.1 * len(rows)))
    for r, (part, note) in enumerate(rows):
        off = json.load(open(os.path.join(ROOT, "runs", "seq_off", "out", part + ".spec.json")))
        on = json.load(open(os.path.join(ROOT, "runs", "seq_on", "out", part + ".spec.json")))
        src = (off.get("outline") or {}).get("source")
        a, b = by["off"][part], by["on"][part]
        win = b["f1"] > a["f1"]
        ax[r, 0].imshow(load(src))
        ax[r, 0].set_title("%s - %s" % (part, note), fontsize=9, loc="left",
                           color=GOOD if win else BAD)
        draw_spec(ax[r, 1], off)
        ax[r, 1].set_title("tracer   F1 %.2f" % a["f1"], fontsize=9)
        draw_spec(ax[r, 2], on)
        ax[r, 2].set_title("model   F1 %.2f" % b["f1"], fontsize=9,
                           color=GOOD if win else BAD)
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s_ in ax[r, c].spines.values():
                s_.set_visible(False)
    fig.suptitle("seqnet on photographs - green rows are its wins on many-curve parts,\n"
                 "red rows are what it does to plain rectangles; net -0.067 F1, and reverted",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(ROOT, out), dpi=80, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
