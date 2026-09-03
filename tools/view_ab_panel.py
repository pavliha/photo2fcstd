import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageOps

from photo2fcstd import overlay, settings
from photo2fcstd.bench import truth_of

PHOTOS = os.path.join(settings.data_dir(), "captured_img")
OLD, NEW = "#b0522f", "#2f8f4e"


def scores(run):
    rows = (l.split() for l in open("runs/%s/results.txt" % run))
    return {f[0]: float(f[2]) for f in rows if len(f) == 3 and f[2][0].isdigit()}


def sources(run):
    out = {}
    for f in glob.glob("runs/%s/out/*.spec.json" % run):
        spec = json.load(open(f))
        src = spec.get("source") or (spec.get("outline") or {}).get("source")
        out[os.path.basename(f).split(".")[0]] = os.path.basename(src) if isinstance(src, str) else None
    return out


def photo(name):
    hits = glob.glob(os.path.join(PHOTOS, "*", name))
    return hits[0] if hits else None


def movers(a, b, sa, sb, want):
    shared = [p for p in set(a) & set(b) if sa.get(p) and sb.get(p)]
    ordered = sorted(shared, key=lambda p: b[p] - a[p])
    return ordered[-want:][::-1] + ordered[:want]


def cell(ax, part, run, src, score, colour, tag):
    img = photo(src)
    ax.imshow(ImageOps.exif_transpose(Image.open(img))) if img else ax.axis("off")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(4)
    ax.set_title("%s  %s\nsolid IoU %.2f" % (tag, src, score), fontsize=8, color=colour)


def draw(parts, a, b, sa, sb, out, old="before", new="after"):
    fig, axes = plt.subplots(len(parts), 4, figsize=(13, 3.3 * len(parts)), squeeze=False)
    fig.patch.set_facecolor("#faf8f5")
    for row, part in zip(axes, parts):
        cell(row[0], part, old, sa[part], a[part], OLD, "before")
        cell(row[2], part, new, sb[part], b[part], NEW, "after")
        for ax, run in ((row[1], old), (row[3], new)):
            cand = "runs/%s/out/%s.stl" % (run, part)
            c = overlay.compare(truth_of(part), cand)
            overlay.draw(ax, c["views"]["face"], "")
        row[0].set_ylabel("part %s\n%+.3f" % (part, b[part] - a[part]), fontsize=9)
    fig.legend(handles=[Patch(color=(0.55, 0.55, 0.55), label="both"),
                        Patch(color=(0.85, 0.15, 0.15), label="truth only"),
                        Patch(color=(0.15, 0.35, 0.85), label="model only")],
               loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Last night: what the two new models changed", fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    fig.savefig(out, dpi=95, facecolor=fig.get_facecolor())
    return out


def main():
    old, new = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("pick_first", "pick_ranker")
    a, b = scores(old), scores(new)
    sa, sb = sources(old), sources(new)
    parts = movers(a, b, sa, sb, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    print(draw(parts, a, b, sa, sb, sys.argv[1] if len(sys.argv) > 1 else "tools/view_ab_panel.png", old, new))


if __name__ == "__main__":
    main()
