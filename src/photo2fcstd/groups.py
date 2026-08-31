import json
import os

import numpy as np
import trimesh

from photo2fcstd.settings import data_dir

EXTENT_MM = 0.5
VOLUME_FRAC = 0.02


def signature(path, extent_mm=EXTENT_MM, volume_frac=VOLUME_FRAC):
    mesh = trimesh.load(path)
    extents = tuple(round(float(x) / extent_mm) for x in sorted(mesh.extents))
    volume = float(abs(mesh.volume))
    magnitude = round(np.log(max(volume, 1e-9)) / volume_frac) if volume > 0 else 0
    return extents + (magnitude,)


def signatures(parts, root=None):
    root = root or os.path.join(data_dir(), "stl_from_step")
    out = {}
    for part in parts:
        path = os.path.join(root, part + ".stl")
        if os.path.exists(path):
            out[part] = signature(path)
    return out


def group_of(parts, root=None):
    sigs = signatures(parts, root)
    ids, groups = {}, {}
    for part in sorted(sigs):
        key = sigs[part]
        ids.setdefault(key, len(ids))
        groups[part] = ids[key]
    return groups


def duplicate_report(groups):
    sizes = {}
    for part, g in groups.items():
        sizes.setdefault(g, []).append(part)
    repeated = {g: p for g, p in sizes.items() if len(p) > 1}
    return {"parts": len(groups), "groups": len(sizes),
            "repeated_groups": len(repeated),
            "parts_in_repeats": sum(len(p) for p in repeated.values()),
            "largest": max((len(p) for p in sizes.values()), default=0),
            "examples": [sorted(p)[:4] for p in list(repeated.values())[:3]]}


def grouped_split(parts, groups, fraction=0.5, seed=0):
    rng = np.random.default_rng(seed)
    unique = sorted({groups[p] for p in parts if p in groups})
    rng.shuffle(unique)
    cut = int(round(len(unique) * fraction))
    left = set(unique[:cut])
    a = [p for p in parts if p in groups and groups[p] in left]
    b = [p for p in parts if p in groups and groups[p] not in left]
    return a, b


def save(groups, path):
    json.dump(groups, open(path, "w"))
    return path
