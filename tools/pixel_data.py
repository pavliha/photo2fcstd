"""Build a dataset of photo crops labelled by which view drew the best sketch.

Silhouette statistics choose the right photo 47% of the time against 33% for chance, and the
shipped selector reaches +0.038 with them. Segmentation throws away everything that actually
encodes foreshortening - shading across a face, specular highlights, the ellipticity of a hole
seen obliquely - so the question is whether pixels carry what the mask does not.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import bench  # noqa: E402
from photo2fcstd.trace import load, segment_photo  # noqa: E402

SIDE = 128


def crop(path):
    """The part, masked out of its background, square, grey, and normalised."""
    import cv2
    img = np.asarray(load(path))
    m = segment_photo(path)
    if m.sum() < 400:
        return None
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = int(max(y1 - y0, x1 - x0) * 0.6) + 1
    a, b = max(cy - half, 0), min(cy + half, img.shape[0])
    c, d = max(cx - half, 0), min(cx + half, img.shape[1])
    g = cv2.cvtColor(img[a:b, c:d], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mm = m[a:b, c:d].astype(np.float32)
    if g.size == 0 or mm.sum() < 200:
        return None
    inside = g[mm > 0]
    g = (g - inside.mean()) / max(inside.std(), 1e-3)
    g = g * mm
    out = cv2.resize(np.stack([g, mm], -1), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    return out.transpose(2, 0, 1).astype(np.float32)


def one(part):
    try:
        paths = bench.photos_of(part)[:3]
        crops = [crop(p) for p in paths]
        if any(c is None for c in crops) or len(crops) < 3:
            return None
        return {"part": part, "x": np.stack(crops)}
    except Exception:
        return None


def main(limit=1200):
    """Crops do not need labels, so build them for every candidate part and join later."""
    import photo2fcstd.sketch_score as SS
    ideal = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p])]
    parts = bench.with_photos(parts)[0][:limit]
    with Pool(5) as pool:
        rows = [r for r in pool.map(one, parts) if r]
    X = np.stack([r["x"] for r in rows])
    np.savez_compressed(os.path.join(ROOT, "data", "pixel_crops.npz"),
                        X=X, parts=np.array([r["part"] for r in rows]))
    print("%d parts with three usable crops, X %s" % (len(rows), X.shape))


if __name__ == "__main__":
    main()
