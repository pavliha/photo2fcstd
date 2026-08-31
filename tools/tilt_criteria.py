"""Can anything visible in the photo pick the warp that the ground truth says is best?

The oracle over a grid of rectifying warps is worth +0.090 of sketch IoU. An estimator has to
choose without the answer, so this records image-only statistics for every warp and asks how
much of that ceiling each one recovers.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod  # noqa: E402
from photo2fcstd.trace import segment_photo, upright_mask  # noqa: E402
from tilt_ceiling import AZIMUTHS, TILTS, unrotate  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def angles_of(loops):
    out = []
    for lp in loops:
        for e in lp.get("elements", []):
            if e["type"] == "line":
                d = np.array(e["p1"]) - np.array(e["p0"])
                if np.hypot(*d) > 1e-6:
                    out.append(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    return np.array(out)


def regularity(loops):
    """How close the straight edges are to a common right-angled frame."""
    a = angles_of(loops)
    if len(a) < 3:
        return 0.0, 0.0
    axis = np.abs(((a[:, None] - a[None, :] + 90) % 180) - 90)
    parallel = float(np.mean(axis < 6.0))
    best = max(float(np.mean(np.minimum(np.abs(((a - o) % 90)), 90 - np.abs(((a - o) % 90))) < 6.0))
               for o in np.arange(0, 90, 2.0))
    return parallel, best


def one(part):
    try:
        masks = [upright_mask(segment_photo(p))[0] for p in bench.photos_of(part)[:3]]
        base = max(masks, key=lambda m: int(m.sum()))
        rows = {}
        for t in TILTS:
            for k in range(AZIMUTHS if t > 0 else 1):
                az = 180.0 * k / AZIMUTHS
                m = unrotate(base, t, az)
                if m.sum() < 400:
                    continue
                try:
                    v = analysis.view_from_mask(m, part)
                    doc = spec_mod.assemble([v], name=part, log=lambda *a: None)
                    loops = (doc.get("outline") or {}).get("loops", [])
                    par, ortho = regularity(loops)
                    rows["%.0f_%.0f" % (t, az)] = {
                        "iou": SS.score_one(doc, IDEAL[part])["region_iou"],
                        "rect": v["shape"]["rectangularity"], "sol": v["shape"]["solidity"],
                        "nel": sum(len(l.get("elements", [])) or 1 for l in loops),
                        "parallel": par, "ortho": ortho,
                        "sym": float(len(v.get("symmetric") or [])),
                    }
                except Exception:
                    continue
        return part, rows
    except Exception:
        return part, None


def main(limit=80):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        res = {k: v for k, v in pool.map(one, parts) if v}
    json.dump(res, open(os.path.join(ROOT, "data", "tilt_criteria.json"), "w"))
    keys = [k for k, v in res.items() if "0_0" in v]
    keen = [k for k in keys if 1 - SS.trivial_score(IDEAL[k]) >= 0.15]
    base = np.mean([res[k]["0_0"]["iou"] for k in keen])
    oracle = np.mean([max(v["iou"] for v in res[k].values()) for k in keen])
    print("%d discriminating parts\n" % len(keen))
    print("  %-26s %8s %10s" % ("chooser", "IoU", "of ceiling"))
    print("  %-26s %8.3f %10s" % ("as shot", base, "-"))
    for name, key, sign in (("most rectangular", "rect", 1), ("most solid", "sol", 1),
                            ("fewest elements", "nel", -1), ("most parallel edges", "parallel", 1),
                            ("most right-angled", "ortho", 1), ("most symmetric", "sym", 1)):
        v = np.mean([res[k][max(res[k], key=lambda g: sign * res[k][g][key])]["iou"] for k in keen])
        print("  %-26s %8.3f %9.0f%%" % (name, v, 100 * (v - base) / max(oracle - base, 1e-9)))
    print("  %-26s %8.3f %9.0f%%" % ("oracle (uses the answer)", oracle, 100))


if __name__ == "__main__":
    main()
