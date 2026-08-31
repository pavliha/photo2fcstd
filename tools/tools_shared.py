import json
import os

import numpy as np

from photo2fcstd import embed
from photo2fcstd.bench import photos_of

MODES = ("stations", "profile", "plan", "revolve")


def vector_for(row):
    found = {os.path.basename(p): p for p in photos_of(row["part"])}
    names = [os.path.basename(v["source"]) for v in sorted(row["views"], key=lambda v: -v["elongation"])]
    paths = [found[n] for n in names if n in found][:3]
    if len(paths) < 3:
        return None
    try:
        got = embed.vectors_for(paths, allow_backbone=False)
    except Exception:
        return None
    return None if got is None else got.reshape(-1)


def mode_data():
    rows = json.load(open("data/mode_labels_full.json"))
    groups = json.load(open("data/part_groups.json"))
    kept, X = [], []
    for row in rows:
        if not row.get("iou_per_mode"):
            continue
        v = vector_for(row)
        if v is None:
            continue
        kept.append(row)
        X.append(v)
    Y = np.array([[r["iou_per_mode"].get(m, np.nan) for m in MODES] for r in kept], float)
    g = np.array([groups.get(r["part"], -1) for r in kept])
    return np.array(X, float), Y, g, kept
