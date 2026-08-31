import json
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

from photo2fcstd import thresholds as th
from photo2fcstd.rectify import PPMM

WARM = 0.005
MIN_STEP_MM = 0.7
STEP_GRAD = 1.6
TAIL = 0.25


def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def largest(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("no object found")
    return lab == int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))) + 1


def segment(a):
    core = largest(ndimage.binary_opening(a.max(axis=2) < 0.32, np.ones((9, 9))))
    ys, xs = np.where(core)
    h, w = core.shape
    pad = int(0.12 * (xs.max() - xs.min()))
    span = ys.max() - ys.min()
    win = np.zeros_like(core)
    win[max(0, ys.min() - int(0.06 * span)):min(h, ys.max() + int(0.70 * span)),
        max(0, xs.min() - pad):min(w, xs.max() + pad)] = True
    m = ((a[..., 0] - a[..., 2]) < WARM) & win
    return largest(ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((13, 13)))))


def otsu(x):
    hist, edges = np.histogram(x, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w1, 1))
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[1:]) ** 2
    return float(mids[int(np.argmax(between))])


def segment_auto(a):
    h, w = a.shape[:2]
    border = np.concatenate([a[:h // 20].reshape(-1, 3), a[-h // 20:].reshape(-1, 3),
                             a[:, :w // 20].reshape(-1, 3), a[:, -w // 20:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    m = dist > otsu(dist)
    m = ndimage.binary_opening(m, np.ones((7, 7)))
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((15, 15))))
    m[:2, :] = m[-2:, :] = m[:, :2] = m[:, -2:] = False
    return largest(m)


_RMBG = {}


from photo2fcstd.settings import cache_dir


def cached_mask(path):
    import hashlib
    st = os.stat(path)
    key = hashlib.sha1(("%s|%d|%d" % (os.path.abspath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()
    return os.path.join(cache_dir("masks"), key + ".npz")


def segment_photo(path):
    f = cached_mask(path)
    if os.path.exists(f):
        return np.load(f)["mask"]
    mask = segment_rmbg(load(path))
    os.makedirs(cache_dir("masks"), exist_ok=True)
    np.savez_compressed(f, mask=mask)
    return mask


def segment_rmbg(a):
    import torch
    from PIL import Image as _Image
    from torchvision import transforms
    if "model" not in _RMBG:
        from transformers import AutoModelForImageSegmentation
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        m = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True).to(dev).eval()
        _RMBG.update(model=m, dev=dev)
    im = _Image.fromarray((a * 255).astype(np.uint8))
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        pred = _RMBG["model"](tf(im).unsqueeze(0).to(_RMBG["dev"]))[-1].sigmoid().cpu()[0, 0]
    alpha = np.asarray(transforms.ToPILImage()(pred).resize(im.size)) > 128
    return largest(alpha)


def segment_any(a, mode="rmbg"):
    return {"dark": segment, "auto": segment_auto}.get(mode, segment_rmbg)(a)


def upright_mask(mask):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    major = vecs[:, int(np.argmax(vals))]
    angle = np.degrees(np.arctan2(major[0], major[1]))
    rotated = ndimage.rotate(mask.astype(np.uint8), -angle, reshape=True, order=0) > 0
    return rotated, float(angle)


def outline(mask, eps_frac=0.008, min_hole=0.002):
    import cv2
    m = mask.astype(np.uint8) * 255
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer_i = max(range(len(cs)), key=lambda i: cv2.contourArea(cs[i]))
    c = cs[outer_i]
    area = cv2.contourArea(c)
    simplify = lambda k: cv2.approxPolyDP(k, eps_frac * cv2.arcLength(k, True), True).reshape(-1, 2).astype(float)
    holes = [simplify(cs[i]) for i in range(len(cs)) if hier[0][i][3] == outer_i and cv2.contourArea(cs[i]) > min_hole * area]
    hole_area = sum(cv2.contourArea(cs[i]) for i in range(len(cs)) if hier[0][i][3] == outer_i)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    x, y, w, h = cv2.boundingRect(c)
    raw_holes = [cs[i].reshape(-1, 2).astype(float).tolist() for i in range(len(cs)) if hier[0][i][3] == outer_i and cv2.contourArea(cs[i]) > min_hole * area]
    return simplify(c), {"rectangularity": float(area / max(w * h, 1)), "stroke_px": float(2 * (area - hole_area) / max(cv2.arcLength(c, True), 1)),
                         "solidity": float(area / max(hull_area, 1)), "bbox": (w, h), "holes": [h_.tolist() for h_ in holes],
                         "hole_frac": float(hole_area / max(area, 1)), "raw": c.reshape(-1, 2).astype(float).tolist(), "raw_holes": raw_holes}

def symmetrize(mask, min_iou=0.93):
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    ys, xs = np.nonzero(filled)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = filled[y0:y1, x0:x1]
    axes = []
    for axis in (1, 0):
        f = np.flip(crop, axis=axis)
        if (crop & f).sum() / max((crop | f).sum(), 1) >= min_iou:
            crop = crop | f
            axes.append("x" if axis == 1 else "y")
    out = filled.copy()
    out[y0:y1, x0:x1] = crop
    return out & ~holes, axes


def fit_circle(pts):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    (cx, cy, c), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x ** 2 + y ** 2, rcond=None)
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-9)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    w, h = np.ptp(x), np.ptp(y)
    return float(cx), float(cy), r, rms / r, float(min(w, h) / max(w, h, 1e-9))


def snap_rectilinear(pts, tol_deg=12.0, lock=None):
    pts = np.asarray(pts, float).copy()
    n = len(pts)
    lock = lock or [False] * n
    for _ in range(3):
        for i in range(n):
            if lock[i]:
                continue
            a, b = pts[i], pts[(i + 1) % n]
            ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180
            if ang < tol_deg or ang > 180 - tol_deg:
                pts[i][1] = pts[(i + 1) % n][1] = (a[1] + b[1]) / 2
            elif abs(ang - 90) < tol_deg:
                pts[i][0] = pts[(i + 1) % n][0] = (a[0] + b[0]) / 2
    return pts


def merge_collinear(pts, min_len, tol_deg=6.0):
    pts = [np.asarray(q, float) for q in pts]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            d1, d2 = b - a, c - b
            turn = abs((np.degrees(np.arctan2(d2[1], d2[0]) - np.arctan2(d1[1], d1[0])) + 180) % 360 - 180)
            if turn < tol_deg or (np.hypot(*d1) < min_len and turn < 45):
                pts.pop(i)
                changed = True
                break
    return np.array(pts)


def arc_span(run, cx, cy):
    ang = np.unwrap(np.arctan2(run[:, 1] - cy, run[:, 0] - cx))
    return float(np.degrees(ang[-1] - ang[0]))


def corner_runs(raw, eps_frac=0.015):
    import cv2
    pts = np.asarray(raw, np.float32)
    poly = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps_frac * cv2.arcLength(pts, True), True).reshape(-1, 2)
    corners = [int(np.argmin(np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1]))) for q in poly]
    corners = sorted(set(corners))
    runs = []
    for j, c in enumerate(corners):
        d = corners[(j + 1) % len(corners)]
        run = pts[c:d + 1] if d > c else np.vstack([pts[c:], pts[:d + 1]])
        runs.append(run.astype(float))
    return runs


MIN_ELLIPSE_POINTS = 12


def ellipse_ok(f, n_points=None):
    if f is None or (n_points is not None and n_points < MIN_ELLIPSE_POINTS):
        return False
    return f["rms"] < max(0.03 * f["b"], 3.0) and f["aspect"] > 0.5


def arc_from_run(run, ccw):
    f = fit_ellipse(run)
    cx, cy, r = (f["cx"], f["cy"], f["a"]) if ellipse_ok(f) else fit_circle(run)[:3]
    c = np.array([cx, cy])
    on = lambda q: (c + (np.asarray(q) - c) / max(np.hypot(*(np.asarray(q) - c)), 1e-9) * r).tolist()
    return {"type": "arc", "p0": on(run[0]), "p1": on(run[-1]), "cx": float(cx), "cy": float(cy), "r": float(r), "ccw": bool(ccw), "_run": run}


def elements(raw, length_px):
    runs = corner_runs(raw)
    els = []
    for run in runs:
        chord = float(np.hypot(*(run[-1] - run[0])))
        if len(run) >= th.ARC_MIN_POINTS and chord > th.ARC_MIN_CHORD_FRAC * length_px:
            cx, cy, r, rel, _ = fit_circle(run)
            span = arc_span(run, cx, cy)
            cv = run[-1] - run[0]
            sag = float(np.max(np.abs(cv[0] * (run[:, 1] - run[0][1]) - cv[1] * (run[:, 0] - run[0][0])) / max(chord, 1e-9)))
            if rel * r < max(th.ARC_FIT_TOL * r, 1.2) and th.ARC_MIN_SPAN_DEG < abs(span) < th.ARC_MAX_SPAN_DEG and sag > th.ARC_MIN_SAG_FRAC * chord:
                els.append(arc_from_run(run, span > 0))
                continue
        els.append({"type": "line", "p0": run[0].tolist(), "p1": run[-1].tolist()})

    def mergeable(a, b):
        if a["type"] != "arc" or b["type"] != "arc" or a["ccw"] != b["ccw"]:
            return False
        joint = np.vstack([a["_run"], b["_run"]])
        return ellipse_ok(fit_ellipse(joint)) or (np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) < 0.10 * a["r"] and abs(a["r"] - b["r"]) < 0.15 * a["r"])

    merged = []
    for e in els:
        if merged and mergeable(merged[-1], e):
            merged[-1] = arc_from_run(np.vstack([merged[-1]["_run"], e["_run"]]), e["ccw"])
        else:
            merged.append(dict(e))
    if len(merged) > 1 and mergeable(merged[-1], merged[0]):
        merged[-1] = arc_from_run(np.vstack([merged[-1]["_run"], merged[0]["_run"]]), merged[0]["ccw"])
        merged.pop(0)
    for e in merged:
        e.pop("_run", None)
    arcs = [i for i, e in enumerate(merged) if e["type"] == "arc"]
    for j, i in enumerate(arcs):
        e = merged[i]
        for k in arcs[:j]:
            o = merged[k]
            if o.get("centre_of") is None and np.hypot(o["cx"] - e["cx"], o["cy"] - e["cy"]) < 0.10 * max(o["r"], e["r"]):
                e["cx"], e["cy"], e["centre_of"] = o["cx"], o["cy"], k
                break
    for e in merged:
        if e["type"] == "arc":
            c = np.array([e["cx"], e["cy"]])
            on = lambda q: (c + (np.array(q) - c) / max(np.hypot(*(np.array(q) - c)), 1e-9) * e["r"]).tolist()
            e["p0"], e["p1"] = on(e["p0"]), on(e["p1"])
    return merged


def regularise_lines(els, length_px):
    pts = np.array([e["p0"] for e in els], float)
    lock = [els[i]["type"] == "arc" or els[i - 1]["type"] == "arc" for i in range(len(els))]
    keep = list(range(len(els)))
    if all(e["type"] == "line" for e in els):
        pts = merge_collinear(snap_rectilinear(pts), 0.015 * length_px)
        pts = snap_rectilinear(pts)
        return [{"type": "line", "p0": pts[i].tolist(), "p1": pts[(i + 1) % len(pts)].tolist()} for i in range(len(pts))]
    pts = snap_rectilinear(pts, lock=lock)
    for i, e in enumerate(els):
        if e["type"] == "arc":
            pts[i], pts[(i + 1) % len(els)] = e["p0"], e["p1"]
    for i, e in enumerate(els):
        e["p0"], e["p1"] = list(map(float, pts[i])), list(map(float, pts[(i + 1) % len(els)]))
    return els


def kinds_of(els):
    out = []
    for e in els:
        if e["type"] != "line":
            out.append("A")
            continue
        d = np.subtract(e["p1"], e["p0"])
        out.append("H" if abs(d[1]) < 1e-6 else "V" if abs(d[0]) < 1e-6 else "F")
    return out


def joins(els, tol_deg=10.0):
    out = []
    n = len(els)
    for i in range(n):
        a, b = els[i - 1], els[i]
        if a["type"] == "line" and b["type"] == "line":
            out.append("")
            continue
        def tangent_dir(e, at_end):
            if e["type"] == "line":
                d = np.subtract(e["p1"], e["p0"])
            else:
                q = np.array(e["p1"] if at_end else e["p0"]); rad = q - np.array([e["cx"], e["cy"]])
                d = np.array([-rad[1], rad[0]]) * (1 if e["ccw"] else -1)
            return d / max(np.hypot(*d), 1e-9)
        da, db = tangent_dir(a, True), tangent_dir(b, False)
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(da, db)), 0, 1)))
        out.append("T" if ang < tol_deg else "P" if abs(ang - 90) < tol_deg else "")
    return out


def fit_ellipse(pts):
    import cv2
    pts = np.asarray(pts, np.float32)
    if len(pts) < 6:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    a, b = max(d1, d2) / 2, min(d1, d2) / 2
    theta = np.radians(ang if d1 >= d2 else ang + 90)
    c, s_ = np.cos(theta), np.sin(theta)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    u, v = x * c + y * s_, -x * s_ + y * c
    rms = float(np.sqrt(np.mean((np.sqrt((u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2) - 1) ** 2)) * (a + b) / 2)
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "theta": float(theta), "rms": rms, "aspect": float(b / max(a, 1e-6))}


def primitives(raw_loops, length_px, circle_aspect=0.7):
    out = []
    outer = fit_ellipse(np.asarray(raw_loops[0], float))
    hole_aspect = 0.7 * outer["aspect"] if ellipse_ok(outer) else 0.5
    for j, raw in enumerate(raw_loops):
        raw = np.asarray(raw, float)
        f = fit_ellipse(raw)
        if ellipse_ok(f) and f["aspect"] > (circle_aspect if j == 0 else hole_aspect):
            out.append({"type": "circle", "cx": f["cx"], "cy": f["cy"], "r": f["a"]})
            continue
        els = regularise_lines(elements(raw, length_px), length_px)
        out.append({"type": "loop", "elements": els, "kinds": kinds_of(els), "joins": joins(els)})
    return out

def concentric_edges(gray, mask, f, r_lo=0.15, r_hi=0.92, min_prom=0.25):
    from scipy.signal import find_peaks
    gy, gx = np.gradient(gray.astype(float))
    inner = ndimage.binary_erosion(mask, iterations=max(3, int(0.03 * f["b"])))
    g = np.hypot(gx, gy) * inner
    c, s_ = np.cos(f["theta"]), np.sin(f["theta"])
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    fr = np.linspace(r_lo, r_hi, 300)
    prof = []
    for t in fr:
        u, v = t * f["a"] * np.cos(ang), t * f["b"] * np.sin(ang)
        x = (f["cx"] + u * c - v * s_).round().astype(int)
        y = (f["cy"] + u * s_ + v * c).round().astype(int)
        ok = (x >= 0) & (x < g.shape[1]) & (y >= 0) & (y < g.shape[0])
        prof.append(g[y[ok], x[ok]].mean() if ok.any() else 0.0)
    prof = np.array(prof)
    prof = np.convolve(prof, np.ones(7) / 7, mode="same")
    prof = prof / max(prof.max(), 1e-9)
    peaks, props = find_peaks(prof, prominence=min_prom)
    order = np.argsort(props["prominences"])[::-1]
    return [float(fr[i]) for i in peaks[order][:2]]


def widths(mask):
    def run(row):
        idx = np.where(row)[0]
        return max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len) if len(idx) else None
    rows = [(y, run(mask[y])) for y in range(mask.shape[0])]
    a = np.array([(y, r[-1] - r[0] + 1) for y, r in rows if r is not None], dtype=float)
    return a[:, 0], a[:, 1]


def trim(y, w, tail=TAIL):
    sm = ndimage.uniform_filter1d(w, 9)
    keep = np.where(sm > tail * sm.max())[0]
    return y[keep.min():keep.max() + 1], w[keep.min():keep.max() + 1]


def stations(z, w, min_step, grad):
    dz = float(np.median(np.abs(np.diff(z)))) or 1.0
    size = max(3, int(round(min_step / dz)))
    sm = ndimage.uniform_filter1d(ndimage.uniform_filter1d(w, size), size)
    g = np.gradient(sm, z)
    edges = [z[0]]
    for i in np.argsort(-np.abs(g)):
        if abs(g[i]) < grad:
            break
        if all(abs(z[i] - e) > min_step for e in edges):
            edges.append(z[i])
    edges = sorted(edges + [z[-1]], reverse=True)
    spans = [(a, b, (z <= a) & (z > b)) for a, b in zip(edges[:-1], edges[1:])]
    return [{"z_from": round(float(a), 3), "z_to": round(float(b), 3),
             "width": round(float(np.median(w[sel])), 3)}
            for a, b, sel in spans if sel.sum() >= 8]


def trace(path, ppmm, min_step, grad, tail, mode="rmbg"):
    mask, _ = upright_mask(segment_any(load(path), mode))
    y, w = trim(*widths(mask), tail=tail)
    z = -(y - y[0]) / ppmm
    return stations(z, w / ppmm, min_step, grad), float(-z.min())


def demo():
    z = -np.arange(0, 30, 0.1)
    w = np.where(z > -10, 12.0, np.where(z > -20, 16.0, 6.0))
    st = stations(z, w, MIN_STEP_MM, STEP_GRAD)
    assert [round(s["width"]) for s in st] == [12, 16, 6], st
    y = np.arange(300, dtype=float)
    prof = np.where(y < 200, 16.0, np.where(y < 260, 5.0, 1.5))
    ty, tw = trim(y, prof)
    assert tw.min() >= 5.0 and abs(len(ty) - 260) <= 6, (len(ty), tw.min())
    print("photo2sketch self-check ok: 3 stations found, thin tail kept, wire cut")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if not args:
        demo()
        raise SystemExit("\nphoto2sketch.py <front.png> [side.png] [--ppmm=20] [--min-step=0.7] "
                         "[--grad=1.6] [--tail=0.25] [--segment=rmbg|auto|dark] [--json=out.json]\n"
                         "images should be rectified by rectify.py; --ppmm skips the two-view check")
    ppmm = float(opt.get("ppmm", PPMM))
    p = (float(opt.get("min-step", MIN_STEP_MM)), float(opt.get("grad", STEP_GRAD)),
         float(opt.get("tail", TAIL)))
    result = {}
    for name, path in zip(("front", "side"), args):
        st, length = trace(path, ppmm, *p, mode=opt.get("segment", "rmbg"))
        result[name] = {"source": path, "length": round(length, 3), "stations": st}
        print("%s: %s  length %.2f mm, %d stations" % (name, path, length, len(st)))
        print("   z from    z to     width")
        print("\n".join("   %7.2f %7.2f   %7.2f" % (s["z_from"], s["z_to"], s["width"]) for s in st))
    if len(result) == 2 and "ppmm" in opt:
        print("\ncross-check SKIPPED: one --ppmm for both views cannot validate scale")
    elif len(result) == 2:
        a, b = result["front"]["length"], result["side"]["length"]
        print("\nlength agreement between views: %.2f vs %.2f mm  (%.1f%%)"
              % (a, b, 100 * abs(a - b) / max(a, b)))
    if "json" in opt:
        with open(opt["json"], "w") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", opt["json"])
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
