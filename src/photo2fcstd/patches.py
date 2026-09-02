import hashlib
import os

import numpy as np

from photo2fcstd import embed
from photo2fcstd.settings import cache_dir

VERSION = "1"
POOL = int(os.environ.get("P2F_PATCH_POOL", 2))


def cached_path(path):
    stat = os.stat(path)
    raw = "%s|%d|%d|%s|p%s|v%s" % (os.path.abspath(path), stat.st_size, int(stat.st_mtime),
                                   embed.MODEL, POOL, VERSION)
    return os.path.join(cache_dir("patches"), hashlib.sha1(raw.encode()).hexdigest() + ".npz")


def grid_of(tokens):
    side = int(round(tokens.shape[0] ** 0.5))
    return tokens[:side * side].reshape(side, side, -1)


def pooled(grid, factor):
    if factor <= 1:
        return grid
    side = grid.shape[0] - grid.shape[0] % factor
    block = grid[:side, :side].reshape(side // factor, factor, side // factor, factor, -1)
    return block.mean(axis=(1, 3))


def compute(paths):
    import torch
    proc, model = embed.backbone()
    device = embed.device()
    images = [embed.crop(p) for p in paths]
    batch = proc(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch).last_hidden_state
    grids = []
    for i in range(out.shape[0]):
        grid = grid_of(out[i, 1:].float().cpu().numpy())
        grids.append(pooled(grid, POOL).astype(np.float16))
    return grids


def tokens_for(paths, allow_backbone=True):
    """A spatial grid of features per photo, cached like the masks are."""
    targets = {p: cached_path(p) for p in paths}
    cold = [p for p, f in targets.items() if not os.path.exists(f)]
    if cold:
        if not allow_backbone:
            return None
        os.makedirs(cache_dir("patches"), exist_ok=True)
        for path, grid in zip(cold, compute(cold)):
            np.savez_compressed(targets[path], grid=grid)
    return [np.load(targets[p])["grid"].astype(np.float32) for p in paths]


def warm(paths, batch=8):
    cold = [p for p in paths if not os.path.exists(cached_path(p))]
    for i in range(0, len(cold), batch):
        tokens_for(cold[i:i + batch])
    return len(cold)
