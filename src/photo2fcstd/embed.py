import hashlib
import os

import numpy as np

from photo2fcstd.settings import cache_dir

MODEL = os.environ.get("P2F_BACKBONE", "facebook/dinov3-vitl16-pretrain-lvd1689m")
SIZE = 224
MARGIN = 0.08
DIMS = 1024
_CACHE = {}


def key_for(path):
    stat = os.stat(path)
    raw = "%s|%d|%d|%s" % (os.path.abspath(path), stat.st_size, int(stat.st_mtime), MODEL)
    return hashlib.sha1(raw.encode()).hexdigest()


def cached_path(path):
    return os.path.join(cache_dir("embeddings"), key_for(path) + ".npy")


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def backbone():
    if "b" not in _CACHE:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(device()).eval()
        _CACHE["b"] = (proc, model)
    return _CACHE["b"]


def crop(path):
    from PIL import Image, ImageOps
    from photo2fcstd.trace import segment_photo
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    mask = np.asarray(segment_photo(path))
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return image.resize((SIZE, SIZE))
    h, w = mask.shape
    sx, sy = image.width / w, image.height / h
    x0, x1, y0, y1 = xs.min() * sx, xs.max() * sx, ys.min() * sy, ys.max() * sy
    pad = MARGIN * max(x1 - x0, y1 - y0)
    box = (max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad))
    return image.crop(box).resize((SIZE, SIZE))


def embed_images(images):
    import torch
    proc, model = backbone()
    batch = proc(images=images, return_tensors="pt").to(device())
    with torch.no_grad():
        out = model(**batch)
    pooled = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()


def vectors_for(paths, allow_backbone=True):
    hits = {p: cached_path(p) for p in paths}
    cold = [p for p, f in hits.items() if not os.path.exists(f)]
    if cold:
        if not allow_backbone:
            from photo2fcstd import fallback
            fallback.note("embed", "%d images uncached and the backbone is disabled" % len(cold))
            return None
        computed = embed_images([crop(p) for p in cold])
        os.makedirs(cache_dir("embeddings"), exist_ok=True)
        for p, v in zip(cold, computed):
            np.save(hits[p], v.astype(np.float32))
    return np.stack([np.load(hits[p]) for p in paths])


def for_views(views, allow_backbone=True):
    ordered = sorted(views, key=lambda v: -v["elongation"])[:3]
    paths = [v["source"] for v in ordered]
    if len(paths) < 3:
        from photo2fcstd import fallback
        fallback.note("embed", "needs three views, given %d" % len(paths))
        return None
    if not all(os.path.exists(p) for p in paths):
        from photo2fcstd import fallback
        fallback.note("embed", "view sources are not readable image paths")
        return None
    got = vectors_for(paths, allow_backbone)
    return None if got is None else got.reshape(-1)


def warm(paths, jobs=1):
    done = 0
    for path in paths:
        if not os.path.exists(cached_path(path)):
            vectors_for([path])
            done += 1
    return done


def main(argv=None):
    import argparse
    from photo2fcstd.bench import photos_of
    p = argparse.ArgumentParser(prog="photo2fcstd-embed")
    p.add_argument("--ids", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    parts = open(a.ids).read().split()
    parts = parts[:a.limit] if a.limit else parts
    paths = [q for part in parts for q in photos_of(part)[:3]]
    cold = [q for q in paths if not os.path.exists(cached_path(q))]
    print("%d photos, %d already cached, embedding %d" % (len(paths), len(paths) - len(cold), len(cold)))
    for i in range(0, len(cold), 8):
        vectors_for(cold[i:i + 8])
        if (i + 8) % 200 < 8:
            print("  %d/%d" % (i + 8, len(cold)), flush=True)
    print("embedding cache warm for %d photos" % len(paths))


def run():
    main()


if __name__ == "__main__":
    main()
