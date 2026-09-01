"""Is the depth pixel path failing because the backbone is out of its depth, or the head?

Embedding norm cannot answer it - DINOv3 vectors are L2 normalised, so every object reads 1.7. The
question a norm was meant to ask is whether T-LESS lands where PrintCAD does, and cosine geometry
answers that directly: how far a T-LESS embedding sits from the PrintCAD centroid, against how far
PrintCAD's own images sit from it.
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import bench, embed, sketch_score as SS, tless  # noqa: E402


def rgb_for(mask_path):
    d = os.path.dirname(os.path.dirname(mask_path))
    stem = os.path.basename(mask_path).split("_")[0]
    got = os.path.join(d, "rgb", stem + ".png")
    return got if os.path.exists(got) else None


def vectors(paths):
    out = []
    for p in paths:
        try:
            v = embed.vectors_for([p], True)
            if v is not None:
                out.append(np.asarray(v, float).reshape(-1))
        except Exception:
            continue
    return np.array(out) if out else np.zeros((0, embed.DIMS))


def main(n_print=60, n_tless=30):
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p])]
    parts = bench.with_photos(parts)[0][:n_print]
    pc = vectors([bench.photos_of(p)[0] for p in parts])
    tl_paths = []
    for obj in range(1, n_tless + 1):
        try:
            pairs = tless.views_of(obj, every=200, limit=2)
            for _, mp in pairs[:1]:
                got = rgb_for(mp)
                if got:
                    tl_paths.append(got)
        except Exception:
            continue
    tl = vectors(tl_paths)
    if len(pc) < 5 or len(tl) < 5:
        print("not enough embeddings: PrintCAD %d, T-LESS %d" % (len(pc), len(tl)))
        return
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    up, ut = unit(pc), unit(tl)
    centre = up.mean(axis=0)
    centre /= max(np.linalg.norm(centre), 1e-9)
    d_in = 1.0 - up @ centre
    d_out = 1.0 - ut @ centre
    print("embedding geometry, PrintCAD n=%d against T-LESS n=%d\n" % (len(pc), len(tl)))
    print("  %-34s %8s %8s" % ("", "PrintCAD", "T-LESS"))
    print("  %-34s %8.2f %8.2f" % ("raw norm (median)", np.median(np.linalg.norm(pc, axis=1)),
                                   np.median(np.linalg.norm(tl, axis=1))))
    print("  %-34s %8.3f %8.3f" % ("cosine distance to PrintCAD centre",
                                   float(np.median(d_in)), float(np.median(d_out))))
    print("  %-34s %8.3f %8.3f" % ("  90th percentile", float(np.percentile(d_in, 90)),
                                   float(np.percentile(d_out, 90))))
    overlap = float(np.mean(d_out <= np.percentile(d_in, 95)))
    print("\n  T-LESS images inside PrintCAD's 95th percentile: %.0f%%" % (100 * overlap))
    print("  %s" % ("the backbone puts them in the same region, so the head is the problem"
                    if overlap > 0.5 else
                    "the backbone puts T-LESS somewhere PrintCAD never goes, so the head was never asked this question"))
    json.dump({"printcad_median": float(np.median(d_in)), "tless_median": float(np.median(d_out)),
               "overlap": overlap}, open(os.path.join(ROOT, "data", "embed_shift.json"), "w"))


if __name__ == "__main__":
    main()
