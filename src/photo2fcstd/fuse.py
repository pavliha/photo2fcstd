import os

import numpy as np


def section_mask(carved, axis=2, canvas=900):
    """The carve's mid-depth cross-section as a clean mask, staircase removed by marching squares."""
    import cv2
    from scipy import ndimage
    from skimage import measure
    pts = carved["points_mm"]
    vox = carved["voxel_mm"]
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    band = pts[np.abs(pts[:, axis] - (lo + hi) / 2) <= vox * 0.6]
    if len(band) < 20:
        return None, None
    keep = [i for i in range(3) if i != axis]
    ij = np.round(band[:, keep] / vox).astype(int)
    ij -= ij.min(axis=0)
    grid = np.zeros(ij.max(axis=0) + 3, np.float32)
    grid[ij[:, 0] + 1, ij[:, 1] + 1] = 1.0
    grid = ndimage.gaussian_filter(grid, 0.8)
    contours = [c for c in measure.find_contours(grid, 0.5) if len(c) >= 8]
    if not contours:
        return None, None
    span = max(grid.shape)
    scale = (canvas * 0.8) / span
    img = np.zeros((canvas, canvas), np.uint8)
    contours.sort(key=lambda c: -cv2.contourArea(np.round(c * scale).astype(np.int32)))
    for rank, c in enumerate(contours):
        q = np.round(c * scale + canvas * 0.1).astype(np.int32)[:, ::-1]
        cv2.fillPoly(img, [q], 0 if rank else 1)
    return img, scale / vox


def measured_depth(carved, axis=2):
    """Median top-surface height over interior columns - the hull's skirt cannot reach them."""
    from scipy import ndimage
    pts = carved["points_mm"]
    vox = carved["voxel_mm"]
    keep = [i for i in range(3) if i != axis]
    ij = np.round(pts[:, keep] / vox).astype(int)
    ij -= ij.min(axis=0)
    occ = np.zeros(ij.max(axis=0) + 1, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    core = ndimage.binary_erosion(occ, iterations=3)
    tops = {}
    for (i, j), z in zip(map(tuple, ij), pts[:, axis]):
        if core[i, j]:
            tops[(i, j)] = max(tops.get((i, j), 0.0), z)
    if not tops:
        return float(np.ptp(pts[:, axis])) + vox
    return float(np.median(list(tops.values())) + vox)


def fused_spec(carved, name="part", axis=2, length_mm=None):
    """A buildable outline spec from a posed multi-view carve: the one path whose depth is measured.

    The drawing comes from the mid-section through the production tracer (measured at parity with
    the photo path); the depth comes from the carve (measured to 1-5% on the gate), so this is the
    first spec whose `depth_trusted` is True on the pipeline's own evidence rather than a guess.
    """
    from photo2fcstd import analysis, spec as spec_mod
    mask, px_per_unit = section_mask(carved, axis)
    if mask is None:
        raise ValueError("the carve has no usable mid-section - check the masks and poses")
    view = analysis.view_from_mask(mask)
    loops = spec_mod.traced_outline(view)
    if not loops:
        raise ValueError("the section traced to no area")
    depth_px = measured_depth(carved, axis) * px_per_unit
    outline = {"source": "fused:%d views" % carved.get("views", 0), "loops": loops,
               "depth_px": depth_px,
               "depth_note": "depth measured by the %d-view carve (median interior top height)"
                             " (px units)" % carved.get("views", 0),
               "depth_trusted": True}
    mpp, scale_note = (length_mm / view["length_px"],
                       "from --length-mm %s over %.1f px" % (length_mm, view["length_px"])) \
        if length_mm else (1.0, "UNSCALED: set this from one caliper reading (mm / px)")
    known = length_mm is not None
    from photo2fcstd import thresholds as th
    q = th.ROUND_MM if known else 1.0
    rnd = lambda v: round(round(v * mpp / q) * q, 4)
    outline["loops"] = spec_mod.rounded_loops(outline["loops"], rnd)
    outline["depth_px"] = rnd(outline["depth_px"])
    return {"name": name, "mode": "fused", "mm_per_px": 1.0,
            "unit": "mm" if known else "px",
            "scale_note": scale_note if known else scale_note + "; the sheet is in pixels until you set scale",
            "views": {}, "outline": outline, "revolve": None, "stl": None, "measured": []}
