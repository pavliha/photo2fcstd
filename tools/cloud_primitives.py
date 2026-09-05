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


def revolve_spec(prof, name, scale=1000.0, frame=None):
    pts = [[0.0, 0.0]] + [[round(r * scale, 2), round(h * scale, 2)] for r, h in prof] + [[0.0, round(prof[-1][1] * scale, 2)]]
    return {"name": name, "mode": "revolve", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: one caliper reading sets scale (cloud units x1000)",
            "views": {}, "outline": None, "stl": None, "measured": [],
            "revolve": {"generic": True, "profile": pts, "holes": [], "rings": []},
            "_cloud_frame": frame}


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
            "features": [{"op": "pad", "depth_px": round(depth_px, 2), "loops": [scale_loop(l) for l in loops]}],
            "_cloud_frame": {"kind": "box", "mn": mn.tolist(), "s": float(s), "factor": float(factor), "lo": float(lo), "scale": scale,
                             "raster": img}}


def contain_fraction(spec, X, Y, H, tol_frac=0.03):
    fr = spec.get("_cloud_frame")
    if not fr:
        return None
    idx = np.random.default_rng(1).choice(len(X), min(len(X), 20000), replace=False)
    x, y, h = X[idx], Y[idx], H[idx]
    if fr["kind"] == "revolve":
        prof = np.array(spec["revolve"]["profile"][1:-1])
        r = np.hypot(x - fr["cx"], y - fr["cy"]) * fr["scale"]
        hh = (h - fr["lo"]) * fr["scale"]
        rmax = np.interp(hh, prof[:, 1], prof[:, 0], left=prof[0, 0], right=prof[-1, 0])
        tol = tol_frac * prof[:, 0].max()
        inside = (r <= rmax + tol) & (hh >= -tol) & (hh <= prof[-1, 1] + tol)
        return float(inside.mean())
    import cv2
    img = fr["raster"].astype(np.uint8)
    k = max(3, int(tol_frac * max(img.shape)))
    img = cv2.dilate(img, np.ones((k, k), np.uint8)) > 0
    q = ((np.column_stack([x, y]) - np.array(fr["mn"])) * fr["s"] + 10).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < img.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < img.shape[0])
    z = (h - fr["lo"]) * fr["scale"]
    depth = spec["features"][0]["depth_px"]
    tol = tol_frac * depth
    inside = np.zeros(len(x), bool)
    inside[ok] = img[q[ok, 1], q[ok, 0]]
    inside &= (z >= -tol) & (z <= depth + tol)
    return float(inside.mean())


def main(npz_path, out_json, name="part"):
    npz = np.load(npz_path)
    X, Y, H = coords(npz)
    kind, stats, prof = classify(X, Y, H)
    lo = float(np.quantile(H, 0.02))
    if kind == "revolve":
        spec = revolve_spec(prof, name, frame={"kind": "revolve", "cx": float(np.median(X)), "cy": float(np.median(Y)), "lo": lo, "scale": 1000.0})
    else:
        spec = box_spec(X, Y, H, name)
    spec["contain_fraction"] = contain_fraction(spec, X, Y, H)
    stats["contain_fraction"] = spec["contain_fraction"]
    if spec.get("_cloud_frame"):
        spec["_cloud_frame"].pop("raster", None)
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


def fit_revolve(npz_path, scale=1000.0):
    npz = np.load(npz_path)
    X, Y, H = coords(npz)
    kind, stats, prof = classify(X, Y, H, bands=24)
    params, ledger = fit_revolve_profile([(p[0] * scale, p[1] * scale) for p in prof], "3D cloud")
    prof_pts = _template_profile(params)
    spec_like = {"revolve": {"profile": prof_pts},
                 "_cloud_frame": {"kind": "revolve", "cx": float(np.median(X)), "cy": float(np.median(Y)),
                                  "lo": float(np.quantile(H, 0.02)), "scale": scale}}
    return params, ledger, {"kind": kind, "contain_fraction": contain_fraction(spec_like, X, Y, H), **stats}


def _template_profile(p):
    h1 = p["body_h"] + p["shoulder_h"]; h2 = h1 + p["neck_h"]; h3 = h2 + p["cap_h"]
    return [[0, 0], [p["body_r"], 0], [p["body_r"], p["body_h"]], [p["neck_r"], h1], [p["neck_r"], h2],
            [p["cap_r"], h2], [p["cap_r"], h3], [0, h3]]


def fit_revolve_profile(prof, source):
    r = np.array([p[0] for p in prof], float)
    h = np.array([p[1] for p in prof], float)
    total = float(h[-1])
    body_r = float(np.median(r[h < 0.5 * total]))
    below = np.where(r < 0.85 * body_r)[0]
    below = below[below > len(r) // 4] if len(below) else below
    i_sh = int(below[0]) if len(below) else len(r) - 2
    body_h = float(h[i_sh - 1]) if i_sh > 0 else 0.5 * total
    upper = r[i_sh:]
    neck_r = float(np.min(upper)) if len(upper) else 0.5 * body_r
    i_neck = i_sh + int(np.argmin(upper)) if len(upper) else i_sh
    shoulder_h = max(float(h[i_neck] - body_h), 0.02 * total)
    cap_r = float(r[-1])
    top = np.where(r[i_neck:] > neck_r * 1.15)[0]
    cap_h = float(h[-1] - h[i_neck + top[0]]) if len(top) else 0.05 * total
    cap_r = float(np.median(r[i_neck + top[0]:])) if len(top) else neck_r
    neck_h = max(total - body_h - shoulder_h - cap_h, 0.02 * total)
    params = {k: round(v, 2) for k, v in dict(body_r=body_r, body_h=body_h, shoulder_h=shoulder_h,
                                              neck_r=neck_r, neck_h=neck_h, cap_r=cap_r, cap_h=cap_h).items()}
    ledger = {k: "measured (%s, %d bands)" % (source, len(r)) for k in params}
    return params, ledger


def build_revolve_template(params, ledger, out):
    import subprocess, tempfile
    pj = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({**params, "_ledger": ledger}, pj); pj.close()
    freecad = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revolve_template.py")
    r = subprocess.run([freecad, tool], env=dict(os.environ, P2F_PARAMS=pj.name, P2F_OUT=out),
                       capture_output=True, text=True, timeout=300)
    if "SAVED" not in r.stdout:
        raise RuntimeError((r.stdout + r.stderr)[-600:])
    return out
