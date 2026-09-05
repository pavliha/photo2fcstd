import json
import os
import sys

import numpy as np


def frame_from(npz):
    n = npz["plane_n"] / np.linalg.norm(npz["plane_n"])
    a0 = npz["object_axes"][0]
    x = a0 - (a0 @ n) * n
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(n, x)
    return x, y, n


def coords(npz):
    P = npz["cloud"].astype(float)
    x, y, n = frame_from(npz)
    d = float(npz["plane_d"])
    H = P @ n + d
    c = P[H > 0].mean(0)
    X, Y = (P - c) @ x, (P - c) @ y
    return X, Y, H


def classify(X, Y, H, bands=12):
    lo, hi = np.quantile(H, 0.02), np.quantile(H, 0.98)
    R = np.hypot(X - np.median(X), Y - np.median(Y))
    ext = np.sort([np.quantile(X, .98) - np.quantile(X, .02), np.quantile(Y, .98) - np.quantile(Y, .02)])[::-1]
    roundness = float(ext[1] / ext[0])
    edges = np.linspace(lo, hi, bands + 1)
    spreads, prof = [], []
    for k in range(bands):
        sel = (H >= edges[k]) & (H < edges[k + 1])
        if sel.sum() < 50:
            continue
        r = R[sel]
        ang = np.arctan2(Y[sel] - np.median(Y), X[sel] - np.median(X))
        rim = np.array([np.quantile(r[(ang >= a) & (ang < a + np.pi / 6)], 0.9)
                        for a in np.linspace(-np.pi, np.pi, 12, endpoint=False)
                        if ((ang >= a) & (ang < a + np.pi / 6)).sum() > 5])
        if len(rim) >= 6:
            spreads.append(float(rim.std() / (rim.mean() + 1e-9)))
        prof.append((float(np.quantile(r, 0.9)), float(edges[k + 1] - lo)))
    rim_spread = float(np.median(spreads)) if spreads else 1.0
    kind = "revolve" if roundness > 0.85 and rim_spread < 0.15 else "box"
    return kind, {"roundness": roundness, "rim_spread": rim_spread, "height": float(hi - lo),
                  "footprint_long": float(ext[0]), "footprint_short": float(ext[1])}, prof


def revolve_spec(prof, name, scale=1000.0):
    pts = [[0.0, 0.0]] + [[round(r * scale, 2), round(h * scale, 2)] for r, h in prof] + [[0.0, round(prof[-1][1] * scale, 2)]]
    return {"name": name, "mode": "revolve", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: one caliper reading sets scale (cloud units x1000)",
            "views": {}, "outline": None, "stl": None, "measured": [],
            "revolve": {"generic": True, "profile": pts, "holes": [], "rings": []}}


def box_spec(X, Y, H, name, scale=1000.0, res=512):
    import cv2
    from photo2fcstd import spec as spec_mod, analysis
    from photo2fcstd.trace import outline, upright_mask
    lo, hi = np.quantile(H, 0.02), np.quantile(H, 0.98)
    top = H > lo + 0.5 * (hi - lo)
    px = np.column_stack([X[top], Y[top]])
    mn, mx = np.quantile(px, 0.01, 0), np.quantile(px, 0.99, 0)
    s = (res - 20) / max(np.ptp(np.vstack([mn, mx]), 0).max(), 1e-9)
    img = np.zeros((res, res), np.uint8)
    q = ((px - mn) * s + 10).astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 0] < res) & (q[:, 1] >= 0) & (q[:, 1] < res)]
    img[q[:, 1], q[:, 0]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    from scipy import ndimage
    lab, k = ndimage.label(img > 0)
    if k > 1:
        img = (lab == (np.argmax(ndimage.sum(img > 0, lab, range(1, k + 1))) + 1)).astype(np.uint8)
    img = ndimage.binary_fill_holes(img > 0)
    view = analysis.view_from_mask(img)
    loops = spec_mod.traced_outline(view)[:1]        # the outer footprint only; interior detail is not in a footprint
    depth_px = float(hi - lo) * scale
    unit_px = (res - 20) / max(np.ptp(np.vstack([mn, mx]), 0).max(), 1e-9)
    factor = scale / unit_px
    def sc(v): return round(float(v) * factor, 2)
    def scale_loop(l):
        if l["type"] == "circle":
            return {**l, "cx": sc(l["cx"]), "cy": sc(l["cy"]), "r": sc(l["r"])}
        els = []
        for e in l["elements"]:
            e2 = {**e, "p0": [sc(e["p0"][0]), sc(e["p0"][1])], "p1": [sc(e["p1"][0]), sc(e["p1"][1])]}
            for k in ("cx", "cy", "r"):
                if k in e: e2[k] = sc(e[k])
            els.append(e2)
        return {**l, "elements": els}
    return {"name": name, "mode": "features", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: one caliper reading sets scale (cloud units x1000)",
            "views": {}, "outline": None, "revolve": None, "stl": None, "measured": [],
            "features": [{"op": "pad", "depth_px": round(depth_px, 2), "loops": [scale_loop(l) for l in loops]}]}


def main(npz_path, out_json, name="part"):
    npz = np.load(npz_path)
    X, Y, H = coords(npz)
    kind, stats, prof = classify(X, Y, H)
    spec = revolve_spec(prof, name) if kind == "revolve" else box_spec(X, Y, H, name)
    json.dump(spec, open(out_json, "w"))
    if kind == "box":
        pts = np.array([p for l in spec["features"][0]["loops"] for e in l["elements"] for p in (e["p0"], e["p1"])])
        ext = np.sort(np.ptp(pts, 0))[::-1] / 1000.0
        stats["trace_long"], stats["trace_short"] = float(ext[0]), float(ext[1])
        stats["trace_short_over_long"] = float(ext[1] / ext[0])
        stats["trace_height_over_long"] = float(stats["height"] / ext[0])
    json.dump({"kind": kind, **stats}, open(os.path.splitext(out_json)[0] + ".stats.json", "w"), indent=1)
    print(json.dumps({"kind": kind, **stats, "spec": out_json}))
    return kind, stats


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "part")
