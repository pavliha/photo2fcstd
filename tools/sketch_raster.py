"""The ideal sketch as a small occupancy raster - the target a decoder can be trained against.

A set of primitives cannot be supervised without deciding which predicted primitive matches
which true one. A raster needs no such correspondence, which is the property that makes this
trainable at all, and the existing tracer turns a raster back into primitives.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

SIZE = int(os.environ.get("P2F_SKETCH_RASTER", 64))
OUT = "data/sketch_rasters.npz"


def polygon_of(record):
    from shapely.geometry import Polygon
    rings = []
    for loop in record.get("loops") or []:
        pts = [q for element in loop for q in element.get("xy", [])]
        if len(pts) >= 4:
            rings.append(np.asarray(pts, float))
    if not rings:
        return None
    shape = Polygon(rings[0])
    if not shape.is_valid:
        shape = shape.buffer(0)
    for hole in rings[1:]:
        try:
            shape = shape.difference(Polygon(hole).buffer(0))
        except Exception:
            pass
    return shape if not shape.is_empty else None


def raster_of(record, size=SIZE):
    """Occupancy on a square grid, the sketch centred and scaled to fill it."""
    import cv2
    shape = polygon_of(record)
    if shape is None:
        return None
    x0, y0, x1, y1 = shape.bounds
    span = max(x1 - x0, y1 - y0)
    if span <= 0:
        return None
    def place(coords):
        pts = np.asarray(coords, float)
        pts = (pts - np.array([(x0 + x1) / 2, (y0 + y1) / 2])) * ((size - 2) / span)
        return np.round(pts + size / 2).astype(np.int32)
    shapes = shape.geoms if hasattr(shape, "geoms") else [shape]
    grid = np.zeros((size, size), np.uint8)
    for piece in shapes:
        cv2.fillPoly(grid, [place(piece.exterior.coords)], 1)
        for hole in piece.interiors:
            cv2.fillPoly(grid, [place(hole.coords)], 0)
    return grid.astype(np.float32)


def main():
    ideal = json.load(open("data/printcad_ideal_sketches_all.json"))
    parts = sorted(set(open("data/tune_subset.txt").read().split())
                   | set(open("data/test_ids.txt").read().split()))
    keys, grids = [], []
    for part in parts:
        record = ideal.get(part)
        if not record:
            continue
        grid = raster_of(record)
        if grid is None or grid.sum() < 4:
            continue
        keys.append(part)
        grids.append(grid)
    np.savez_compressed(OUT, parts=np.array(keys), rasters=np.stack(grids).astype(np.uint8))
    print("wrote %s: %d sketches at %dx%d, mean fill %.3f"
          % (OUT, len(keys), SIZE, SIZE, float(np.mean(grids))))


if __name__ == "__main__":
    main()
