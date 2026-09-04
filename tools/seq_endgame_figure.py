"""The seqnet ratchet in pictures: what the model proposes, and what the gates let through."""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["P2F_SEQ_DEVICE"] = "cpu"
from chain_figure import GOOD, BAD, DIM, draw_spec  # noqa: E402

PARTS = [("00357", "a win the gates keep: F1 0.68 -> 0.81"),
         ("00082", "small proposals kept, the shattering ones vetoed"),
         ("00189", "the square the v2 model shattered into 25 arcs"),
         ("00548", "the plate whose quad the model bent to IoU 0.30")]


def spec_of(part, seq, gates=True):
    import photo2fcstd.trace as trace
    from photo2fcstd import analysis, bench, spec as spec_mod
    trace.SEQNET = seq
    if not gates:
        os.environ["P2F_SEQ_GATES"] = "0"
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        return spec_mod.assemble(views, name=part, log=lambda *a: None)
    finally:
        os.environ.pop("P2F_SEQ_GATES", None)
        trace.SEQNET = False


def main(out="docs/figures/seq_endgame.png"):
    from photo2fcstd import bench
    from photo2fcstd.sketch_score import counts_of_spec
    from photo2fcstd.trace import load
    fig, ax = plt.subplots(len(PARTS), 4, figsize=(15, 3.3 * len(PARTS)))
    for r, (part, note) in enumerate(PARTS):
        tracer = spec_of(part, seq=False)
        raw = spec_of(part, seq=True, gates=False)
        gated = spec_of(part, seq=True, gates=True)
        src = (tracer.get("outline") or {}).get("source") or bench.photos_of(part)[0]
        ax[r, 0].imshow(load(src))
        ax[r, 0].set_title("%s\n%s" % (part, note), fontsize=9, loc="left")
        raw_label = ("model, ungated" if counts_of_spec(raw) != counts_of_spec(tracer)
                     else "model proposal unusable -> tracer")
        gated_label = ("gated composite: model kept" if counts_of_spec(gated) != counts_of_spec(tracer)
                       else "gated composite: tracer kept")
        for c, (doc, label, col) in enumerate(((tracer, "tracer", DIM),
                                               (raw, raw_label, BAD),
                                               (gated, gated_label, GOOD)), start=1):
            draw_spec(ax[r, c], doc)
            ax[r, c].set_title(label, fontsize=9, color=col)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s_ in ax[r, c].spines.values():
                s_.set_visible(False)
    fig.suptitle("seqnet endgame with the wobble-trained model - middle: the model's own proposal;\n"
                 "right: what the geometric gates let through (net +0.001, zero)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(ROOT, out), dpi=80, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
