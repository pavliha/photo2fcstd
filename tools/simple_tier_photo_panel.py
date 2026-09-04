"""Photo, traced sketch and ideal side by side for the simple parts that score worst."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, "src")
from photo2fcstd.bench import sketch_scores
from photo2fcstd.gallery import draw_sketch
from photo2fcstd.settings import data_dir
from photo2fcstd.sketch_score import ideal_rings

IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/review_check"
PARTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["01348", "00194", "00030", "00123"]
OUT = "docs/figures/simple_tier_photos.png"


def photo_of(part):
    import glob
    hits = sorted(glob.glob(os.path.join(data_dir(), "captured_img", "**", part + "_*.jpg"), recursive=True))
    return hits[0] if hits else None


def run():
    rows = sketch_scores(RUN)
    fig, axes = plt.subplots(3, len(PARTS), figsize=(3.0 * len(PARTS), 8.4))
    for column, part in enumerate(PARTS):
        f1 = (rows.get(part) or {}).get("primitive_f1") or {}
        shot, drawn, ideal = axes[0][column], axes[1][column], axes[2][column]

        path = photo_of(part)
        if path:
            shot.imshow(ImageOps.exif_transpose(Image.open(path)))
        shot.set_title("%s  photo" % part, fontsize=9)

        spec_path = os.path.join(RUN, "out", part + ".spec.json")
        if os.path.exists(spec_path):
            draw_sketch(drawn, json.load(open(spec_path)))
        drawn.set_title("we drew %s" % f1.get("drawn"), fontsize=9)

        for ring in ideal_rings(IDEAL.get(part) or {}):
            if len(ring):
                closed = np.vstack([ring, ring[:1]])
                ideal.plot(closed[:, 0], closed[:, 1], "-", lw=1.6, color="#2f8f4e")
        ideal.set_title("ideal %s   F1 %.2f" % (f1.get("wanted"), f1.get("f1", 0)), fontsize=9)

        for ax in (shot, drawn, ideal):
            ax.set_aspect("equal")
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    run()
