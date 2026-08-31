"""How many real photos give us a round feature good enough to read the tilt off?"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import bench, sketch_score as SS  # noqa: E402
from photo2fcstd.trace import outline, segment_photo, upright_mask  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
MIN_POINTS = 12


def ellipse_quality(pts):
    """Fit an ellipse to a hole and say how well it fits, in units of its own size."""
    p = np.asarray(pts, np.float32)
    if len(p) < MIN_POINTS:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(p)
    a, b = max(MA, ma) / 2, min(MA, ma) / 2
    if b < 3:
        return None
    t = np.radians(ang)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    q = (p - [cx, cy]) @ R.T
    rms = float(np.sqrt(np.mean((np.hypot(q[:, 0] / max(ma / 2, 1e-6), q[:, 1] / max(MA / 2, 1e-6)) - 1) ** 2)))
    return {"a": a, "b": b, "aspect": b / a, "rms": rms,
            "tilt_deg": float(np.degrees(np.arccos(np.clip(b / a, 0, 1))))}


def one(part):
    try:
        photos = bench.photos_of(part)[:3]
        best = None
        for p in photos:
            mask, _ = upright_mask(segment_photo(p))
            _, shape = outline(mask)
            for h in shape.get("raw_holes", []):
                q = ellipse_quality(h)
                if q and q["rms"] < 0.10 and (not best or q["a"] > best["a"]):
                    best = q
        return part, best
    except Exception:
        return part, None


def main(limit=250):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    json.dump({k: v for k, v in out.items()}, open(os.path.join(ROOT, "data", "tilt_feasibility.json"), "w"))
    have = [v for v in out.values() if v]
    print("%d trusted parts with photos\n" % len(out))
    print("  %-42s %5d  %3.0f%%" % ("have a round hole that fits an ellipse", len(have),
                                    100 * len(have) / len(out)))
    for cut in (0.05, 0.03):
        n = [v for v in have if v["rms"] < cut]
        print("  %-42s %5d  %3.0f%%" % ("  ... fitting to within %.0f%% of its size" % (100 * cut),
                                        len(n), 100 * len(n) / len(out)))
    big = [v for v in have if v["a"] >= 12]
    print("  %-42s %5d  %3.0f%%" % ("  ... and at least 12 px across", len(big),
                                    100 * len(big) / len(out)))
    if have:
        t = np.array([v["tilt_deg"] for v in have])
        print("\n  implied tilt: median %.1f deg, quartiles %.1f / %.1f" % (np.median(t), *np.percentile(t, [25, 75])))
        print("  (8 degrees of tilt costs 0.11 of silhouette IoU)")


if __name__ == "__main__":
    main()
