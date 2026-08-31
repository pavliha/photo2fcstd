"""Embed each part's photos with a frozen backbone, cropped to the part."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

from photo2fcstd.bench import photos_of
from photo2fcstd.trace import segment_photo

MODEL = os.environ.get("P2F_BACKBONE", "facebook/dinov3-vitl16-pretrain-lvd1689m")
OUT = "data/depth_embeddings.npz"
SIZE = 224
MARGIN = 0.08


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def crop(path):
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    mask = segment_photo(path)
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return image.resize((SIZE, SIZE))
    h, w = np.asarray(mask).shape
    sx, sy = image.width / w, image.height / h
    x0, x1, y0, y1 = xs.min() * sx, xs.max() * sx, ys.min() * sy, ys.max() * sy
    pad = MARGIN * max(x1 - x0, y1 - y0)
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(image.width, x1 + pad), min(image.height, y1 + pad))
    return image.crop(box).resize((SIZE, SIZE))


def loader():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device()).eval()
    return proc, model


def embed(proc, model, images):
    batch = proc(images=images, return_tensors="pt").to(device())
    with torch.no_grad():
        out = model(**batch)
    pooled = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()


def ordered_photos(row):
    by_elong = sorted(row["views"], key=lambda v: -v["elongation"])
    names = [os.path.basename(v["source"]) for v in by_elong]
    found = {os.path.basename(p): p for p in photos_of(row["part"])}
    return [found[n] for n in names if n in found]


def main(limit=None):
    rows = json.load(open("data/depth_rows.json"))
    rows = rows[:limit] if limit else rows
    proc, model = loader()
    parts, vectors = [], []
    for i, row in enumerate(rows):
        paths = ordered_photos(row)[:3]
        if len(paths) < 3:
            continue
        try:
            vecs = embed(proc, model, [crop(p) for p in paths])
        except Exception as exc:
            print("skip %s: %s" % (row["part"], exc), file=sys.stderr)
            continue
        parts.append(row["part"])
        vectors.append(vecs.reshape(-1))
        if (i + 1) % 50 == 0:
            print("  embedded %d/%d" % (i + 1, len(rows)), file=sys.stderr)
    X = np.stack(vectors)
    np.savez_compressed(OUT, parts=np.array(parts), X=X.astype(np.float32))
    print("wrote %s: %d parts x %d dims" % (OUT, X.shape[0], X.shape[1]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
