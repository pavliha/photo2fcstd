"""Every photograph of one part, what each traces to, and which the selector took."""
import json, os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gallery_photo_to_sketch import draw_rings, draw_spec, frame, INK  # noqa: E402
from photo2fcstd import analysis, bench, modes, sketch_score as SS, spec as spec_mod, view_model  # noqa: E402
from photo2fcstd.trace import load  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def main(part="00911"):
    photos = bench.photos_of(part)[:3]
    views = [analysis.view(p) for p in photos]
    chosen = view_model.choose(views)
    fig, ax = plt.subplots(3, len(photos) + 1, figsize=(4.0 * (len(photos) + 1), 11), squeeze=False)
    best = None
    for i, (p, v) in enumerate(zip(photos, views)):
        doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
        iou = SS.score_one(doc, IDEAL[part])["region_iou"]
        best = i if best is None or iou > best[1] else best
        best = (i, iou) if not isinstance(best, tuple) or iou > best[1] else best
        tag = []
        if i == chosen:
            tag.append("the model chose this")
        colour = "#1a7f37" if i == chosen else "#666"
        ax[0, i].imshow(load(p))
        ax[0, i].set_xticks([]); ax[0, i].set_yticks([])
        ax[0, i].set_title("%s\n%s" % (os.path.basename(p), ", ".join(tag) or " "),
                           fontsize=9, color=colour, fontweight="bold" if tag else "normal")
        mask, _ = analysis.upright_mask(analysis.segment_photo(p))
        ax[1, i].imshow(mask, cmap="gray_r")
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])
        ax[1, i].set_title("silhouette   rectangularity %.2f" % v["shape"]["rectangularity"], fontsize=9)
        counts = draw_spec(ax[2, i], doc)
        frame(ax[2, i], "IoU %.2f   %s" % (iou, counts), colour)
        ax[2, i].relim(); ax[2, i].autoscale()
    for r in range(3):
        ax[r, -1].axis("off")
    draw_rings(ax[2, -1], SS.ideal_rings(IDEAL[part]), INK)
    ax[2, -1].axis("on")
    frame(ax[2, -1], "the sketch the part has")
    ax[0, -1].text(0.0, 0.9, "part %s" % part, fontsize=13, fontweight="bold", va="top")
    ax[0, -1].text(0.0, 0.74,
                   "the mask is a correct silhouette\nof the box as photographed.\n\n"
                   "every view shows top and sides,\nso every outline is a 3D profile\n"
                   "rather than the base face.",
                   fontsize=9.5, va="top")
    fig.suptitle("All three photographs of one part: the silhouette is right, the viewpoint is not",
                 fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(ROOT, "docs", "figures", "one_part_views.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)
    print("per-view IoU:", [round(SS.score_one(spec_mod.assemble([v], name=part, log=lambda *a: None),
                                               IDEAL[part])["region_iou"], 3) for v in views])
    print("model chose index", chosen)


if __name__ == "__main__":
    main(*sys.argv[1:])
