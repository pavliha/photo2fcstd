"""Estimate the rectifying warp from monocular depth instead of from the silhouette."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, "src")
from photo2fcstd import bench, stats
from photo2fcstd.trace import segment_photo, upright_mask

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_M = {}


def model():
    if "m" not in _M:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _M["p"] = AutoImageProcessor.from_pretrained(MODEL)
        _M["m"] = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
        _M["dev"] = dev
    return _M["p"], _M["m"], _M["dev"]


def depth_of(path):
    proc, net, dev = model()
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    inputs = proc(images=image, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = net(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=image.size[::-1], mode="bicubic",
                                        align_corners=False)[0, 0].cpu().numpy()
    return d, np.asarray(image)


def plane_tilt(path):
    d, _ = depth_of(path)
    mask = segment_photo(path)
    if mask.shape != d.shape:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        return None
    keep = np.random.default_rng(0).choice(len(xs), size=min(20000, len(xs)), replace=False)
    ys, xs = ys[keep], xs[keep]
    z = d[ys, xs]
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    A = np.column_stack([(xs - xs.mean()) / span, (ys - ys.mean()) / span, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(A, (z - z.mean()) / max(z.std(), 1e-6), rcond=None)
    gx, gy = float(coef[0]), float(coef[1])
    return {"gx": gx, "gy": gy, "slope": float(np.hypot(gx, gy)),
            "azimuth": float(np.degrees(np.arctan2(gy, gx)) % 180.0)}


def largest_photo(part):
    paths = bench.photos_of(part)[:3]
    if len(paths) < 3:
        return None
    return max(((int(upright_mask(segment_photo(p))[0].sum()), p) for p in paths))[1]


def main(limit=200):
    warps = json.load(open("data/tilt_criteria.json"))
    rows = []
    for i, (part, table) in enumerate(sorted(warps.items())[:limit]):
        if "0_0" not in table:
            continue
        best_key = max(table, key=lambda k: table[k]["iou"])
        tilt, az = (float(x) for x in best_key.split("_"))
        path = largest_photo(part)
        if path is None:
            continue
        est = plane_tilt(path)
        if est is None:
            continue
        rows.append({"part": part, "tilt": tilt, "az": az, "gain": table[best_key]["iou"] - table["0_0"]["iou"],
                     **est})
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, limit), file=sys.stderr)
    json.dump(rows, open("data/tilt_depth.json", "w"))
    moved = [r for r in rows if r["tilt"] > 0 and r["gain"] > 0.02]
    print("parts %d, of which the oracle wants a real warp: %d" % (len(rows), len(moved)))
    if moved:
        err = [min(abs(r["azimuth"] - r["az"]), 180 - abs(r["azimuth"] - r["az"])) for r in moved]
        m, lo, hi = stats.mean_ci(err)
        print("azimuth error: median %.0f deg, mean %.0f [%.0f, %.0f]  (chance is 45)"
              % (np.median(err), m, lo, hi))
        corr = np.corrcoef([r["slope"] for r in moved], [r["tilt"] for r in moved])[0, 1]
        print("slope vs oracle tilt magnitude: pearson %.3f" % corr)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
