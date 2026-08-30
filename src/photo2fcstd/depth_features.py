import glob
import json
import os
import sys

import numpy as np
import trimesh

from photo2fcstd.trace import fit_ellipse, outline, segment_photo, symmetrize, upright_mask

from photo2fcstd.settings import data_dir

D = data_dir()


def view_features(path):
    mask, _ = upright_mask(segment_photo(path))
    mask, _ = symmetrize(mask)
    poly, sh = outline(mask)
    f = fit_ellipse(np.array(sh["raw"]))
    w, h = sh["bbox"]
    return {"min_over_max": min(w, h) / max(w, h), "rect": sh["rectangularity"], "solidity": sh["solidity"],
            "hole_frac": sh["hole_frac"], "aspect": f["aspect"] if f else 1.0, "rms": (f["rms"] / f["b"]) if f else 1.0,
            "area_frac": float(mask.sum()) / (w * h), "diag": float(np.hypot(w, h))}


def main(ids_path, out):
    ids = open(ids_path).read().split()
    rec = {}
    for n, k in enumerate(ids):
        try:
            views = [view_features(p) for p in sorted(glob.glob(f"{D}/captured_img/*/{k}_*.jpg"))]
            e = sorted(trimesh.load(f"{D}/stl_from_step/{k}.stl").extents)
            rec[k] = {"views": views, "t_over_l": e[0] / e[2], "mid_over_l": e[1] / e[2]}
        except Exception as exc:
            rec[k] = {"error": str(exc)[:80]}
        if (n + 1) % 25 == 0:
            print("%d/%d" % (n + 1, len(ids)), flush=True)
    json.dump(rec, open(out, "w"))
    print("wrote", out, len(rec))


def cli(argv):
    main(*argv)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/printcad_test_ids.txt",
         sys.argv[2] if len(sys.argv) > 2 else "data/depth_features.json")
