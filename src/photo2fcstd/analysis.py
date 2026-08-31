import numpy as np

from photo2fcstd import thresholds as th
from photo2fcstd.trace import (MIN_ELLIPSE_POINTS, STEP_GRAD, TAIL, concentric_edges, fit_ellipse, load, outline,
                               segment_any, segment_photo, stations, symmetrize, trim, upright_mask, widths)


def rings_of(img, mask, angle, ellipse):
    from scipy import ndimage
    gray = ndimage.rotate(np.asarray(img, float).mean(axis=2), -angle, reshape=True, order=1)
    return concentric_edges(gray, mask, ellipse) if gray.shape == mask.shape else []


def view(path, min_step_px=None, grad=STEP_GRAD, tail=TAIL, segment="rmbg", rectify=False):
    img = load(path)
    scale = None
    if rectify:
        from photo2fcstd.capture import rectified
        out = rectified(img)
        if out is not None:
            img = out["image"]
            scale = out["mm_per_px"]
            mask, angle = upright_mask(segment_any(img, "rmbg" if segment == "rmbg" else segment))
    if scale is None:
        mask, angle = upright_mask(segment_photo(path) if segment == "rmbg" else segment_any(img, segment))
    mask, sym_axes = symmetrize(mask)
    poly, shape = outline(mask)
    raw = np.array(shape["raw"])
    f = fit_ellipse(raw)
    enough = len(raw) >= MIN_ELLIPSE_POINTS
    shape["round"] = bool(f and enough and f["rms"] < th.ROUND_RMS * f["b"] and f["aspect"] > th.ROUND_ASPECT)
    shape["roundish"] = bool(f and enough and f["rms"] < th.ROUNDISH_RMS * f["b"] and f["aspect"] > th.ROUNDISH_ASPECT)
    shape["ellipse_rms"] = float(f["rms"] / f["b"]) if f else 1.0
    if shape["round"] or shape["roundish"]:
        shape["ellipse"] = f
        shape["rings"] = rings_of(img, mask, angle, f)
    y, w = trim(*widths(mask), tail=tail)
    z = (y - y[0]) * -1.0
    st = stations(z, w, min_step_px or max(8.0, 0.04 * len(w)), grad)
    zmax = float(-z.min())
    up = list(reversed(st))
    edges = [round(zmax + up[0]["z_to"], 3)] + [round(zmax + s["z_from"], 3) for s in up]
    widths_up = [{"width": s["width"]} for s in up]
    elongation = round(zmax / max(max(s["width"] for s in widths_up), 1.0), 2)
    return {"source": path, "angle_deg": round(angle, 1), "length_px": zmax, "poly": poly.tolist(),
            "shape": shape, "symmetric": sym_axes, "stations": widths_up, "z": edges, "elongation": elongation, "mm_per_px": scale}
