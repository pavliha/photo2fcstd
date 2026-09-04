"""MEASURED NULL - the screen for render-and-compare, and it says the signal is not there.

On 92 parts with STEP-recorded depth: choosing the depth whose extruded prism best explains the
other two photographs' silhouettes (16 orientations, scale-free, canonicalised, flip-tested) gives
a median absolute log error of 1.74 against the shipped predictor's 0.24 and a constant's 1.13 -
it loses to the constant, and the per-part deficit against shipped is +1.69 [+1.39, +1.99].
Restricting to parts where the consistency curve actually moves makes it *worse* (+1.79), so the
movement is bias, not signal. This is the edge-on-estimator result re-derived the expensive way:
PrintCAD's three photographs are same-face dominated and depth is not in them. Test-time
render-and-compare needs the sixteen-view capture - it does not work on the archive.


For each candidate depth, extrude the traced front silhouette into a prism, render its
silhouette over a grid of orientations, and score how well the best orientation explains
each of the other two photographs' masks. Training-free; the depth whose prism explains
the other views best wins. Compared against the best constant and the shipped predictor's
own ratio, on parts whose STEP file records the true depth.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RATIOS = np.exp(np.linspace(np.log(0.03), np.log(2.0), 16))
CANVAS = 220


def prism(poly, depth):
    import trimesh
    from shapely.geometry import Polygon
    p = Polygon(poly)
    if not p.is_valid:
        p = p.buffer(0)
    return trimesh.creation.extrude_polygon(p, depth)


def silhouette(mesh, rx, ry):
    import trimesh
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(rx, [1, 0, 0]))
    m.apply_transform(trimesh.transformations.rotation_matrix(ry, [0, 1, 0]))
    v2 = m.vertices[:, :2]
    lo, hi = v2.min(0), v2.max(0)
    span = max((hi - lo).max(), 1e-9)
    q = ((v2 - lo) / span * (CANVAS - 20) + 10).astype(np.int32)
    img = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.fillPoly(img, [q[f] for f in m.faces], 1)
    return img > 0


def upright(mask):
    m = mask.astype(np.uint8)
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(c, key=cv2.contourArea)
    angle = cv2.minAreaRect(big)[2]
    M = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), angle, 1.0)
    pad = int(np.hypot(*m.shape) - min(m.shape)) // 2 + 2
    m = np.pad(m, pad)
    M[0, 2] += pad; M[1, 2] += pad
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST) > 0


def norm_mask(mask):
    mask = upright(mask)
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    s = (CANVAS - 20) / max(crop.shape)
    out = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)),
                     interpolation=cv2.INTER_NEAREST)
    img = np.zeros((CANVAS, CANVAS), np.uint8)
    y0 = (CANVAS - out.shape[0]) // 2
    x0 = (CANVAS - out.shape[1]) // 2
    img[y0:y0 + out.shape[0], x0:x0 + out.shape[1]] = out
    return img > 0


def centred_iou(a, b):
    best = 0.0
    for bb in (b, b[:, ::-1]):
        inter = (a & bb).sum()
        u = (a | bb).sum()
        best = max(best, inter / max(u, 1))
    return best


def explain(mesh, target):
    best = 0.0
    for rx in np.radians([0, 30, 60, 90]):
        for ry in np.radians([0, 30, 60, 90]):
            s = norm_mask(silhouette(mesh, rx, ry))
            best = max(best, centred_iou(s, target))
    return best


def one(part):
    from photo2fcstd import bench
    from photo2fcstd.trace import outline, segment_photo, upright_mask
    try:
        spec = json.load(open(os.path.join(ROOT, "runs", "chain_on", "out", part + ".spec.json")))
        ol = spec.get("outline")
        if not ol:
            return part, None
        src = ol["source"]
        others = [p for p in bench.photos_of(part)[:3] if p != src]
        if len(others) < 2:
            return part, None
        mask, _ = upright_mask(segment_photo(src))
        poly, sh = outline(mask)
        length = max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1]))
        targets = []
        for p in others:
            m, _ = upright_mask(segment_photo(p))
            targets.append(norm_mask(m))
        scores = []
        for r in RATIOS:
            mesh = prism(poly, r * length)
            scores.append(sum(explain(mesh, t) for t in targets))
        chosen = float(RATIOS[int(np.argmax(scores))])
        shipped = float(ol["depth_px"]) / float(spec["views"]["front"]["length_px"])
        return part, {"chosen": chosen, "shipped": shipped, "curve": [float(s) for s in scores]}
    except Exception as e:
        return part, {"error": str(e)[:70]}


def main(limit=90):
    from photo2fcstd import bench
    rows = json.load(open(os.path.join(ROOT, "data", "depth_rows.json")))
    truth = {r["part"]: r["ratio"] for r in rows}
    have = [p for p in sorted(truth) if os.path.exists(os.path.join(ROOT, "runs", "chain_on", "out", p + ".spec.json"))]
    parts = have[:limit]
    with Pool(6) as pool:
        out = dict(pool.map(one, parts))
    ok = {p: v for p, v in out.items() if v and "chosen" in v}
    print("scored %d of %d parts" % (len(ok), len(parts)))
    def spread_of(v):
        c = np.array(v["curve"])
        return float(c.max() - c.min())
    decisive = {p: v for p, v in ok.items() if spread_of(v) >= 0.06}
    print("decisive (curve spread >= 0.06): %d of %d - on the rest depth is not visible to consistency" % (len(decisive), len(ok)))
    from photo2fcstd import stats
    for label, sel in (("all scored", ok), ("decisive only", decisive)):
        if len(sel) < 5:
            continue
        t = np.array([truth[p] for p in sel])
        err = lambda pred: np.abs(np.log(pred / t))
        chosen = np.array([sel[p]["chosen"] for p in sel])
        shipped = np.array([sel[p]["shipped"] for p in sel])
        const = np.exp(np.median(np.log(t)))
        print("\n  [%s, n=%d]" % (label, len(sel)))
        print("  %-34s %-10s %s" % ("predictor", "median", "within 2x"))
        for name, pred in (("consistency (this screen)", chosen), ("shipped pipeline ratio", shipped),
                           ("best constant (oracle median)", np.full_like(t, const))):
            e = err(pred)
            print("  %-34s %-10.3f %.0f%%" % (name, np.median(e), 100 * np.mean(e < np.log(2))))
        m, lo, hi = stats.mean_ci(err(chosen) - err(shipped))
        print("  consistency minus shipped, per part: %+.3f [%+.3f, %+.3f]" % (m, lo, hi))
    json.dump({p: dict(v, true=truth[p]) for p, v in ok.items()},
              open(os.path.join(ROOT, "data", "depth_consistency.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
