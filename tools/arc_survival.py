"""Which ground-truth arcs does the tracer reproduce, and which does it break into lines?

The tracer recovers 1.76 curved primitives against a real 5.19 even given a perfect drawing, so the
arcs are lost in code rather than in capture. This says *which* arcs, because every fix proposed so
far has been a guess about the mechanism.

Each ideal element is matched to the drawn primitives that cover it, in the pixel frame of the
rasterised face, and recorded against the things a fix might plausibly depend on: sweep, radius,
length on the page, and how much of the loop it occupies.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import sketch_score as SS  # noqa: E402
import tracer_ceiling as TC  # noqa: E402

CURVED = ("arc", "circle", "ellipse", "bsplinecurve")


def drawn_points(element, steps=24):
    if element["type"] == "line":
        p0, p1 = np.array(element["p0"], float), np.array(element["p1"], float)
        t = np.linspace(0, 1, steps)[:, None]
        return p0 + t * (p1 - p0)
    c = np.array([element["cx"], element["cy"]], float)
    r = element["r"]
    if element["type"] == "circle":
        a = np.linspace(0, 2 * np.pi, 4 * steps)
    else:
        a0 = np.arctan2(element["p0"][1] - c[1], element["p0"][0] - c[0])
        a1 = np.arctan2(element["p1"][1] - c[1], element["p1"][0] - c[0])
        if element.get("ccw", True) and a1 < a0:
            a1 += 2 * np.pi
        if not element.get("ccw", True) and a1 > a0:
            a1 -= 2 * np.pi
        a = np.linspace(a0, a1, steps)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)


def transform_ideal(record):
    """The same mapping `tracer_ceiling.rasterise` uses, so ideal points land in the drawn frame."""
    loops = [np.concatenate([TC.polyline(e) for e in loop]) for loop in record["loops"]]
    allp = np.concatenate(loops)
    lo, hi = allp.min(0), allp.max(0)
    span = max(hi - lo)
    scale = (TC.SIZE * 0.8) / span
    return lambda p: (np.asarray(p, float) - lo) * scale + TC.SIZE * 0.1, scale


def align(drawn, ideal_px):
    """Bounding-box alignment, trying a y flip, because the traced frame is centred and may invert."""
    best = None
    di = np.concatenate(drawn)
    for flip in (1.0, -1.0):
        pts = di.copy()
        pts[:, 1] *= flip
        lo_d, hi_d = pts.min(0), pts.max(0)
        lo_i, hi_i = ideal_px.min(0), ideal_px.max(0)
        sc = (hi_i - lo_i) / np.maximum(hi_d - lo_d, 1e-9)
        s = float(np.mean(sc))
        shift = lo_i - lo_d * s
        moved = pts * s + shift
        cost = float(np.mean(np.min(np.linalg.norm(
            moved[:, None, :] - ideal_px[None, ::7, :], axis=2), axis=1)))
        if best is None or cost < best[0]:
            best = (cost, flip, s, shift)
    return best


def one(part):
    from photo2fcstd import analysis, spec as spec_mod
    rec = TC.IDEAL[part]
    try:
        mask = TC.rasterise(rec)
        if mask is None or mask.sum() < 500:
            return []
        view = analysis.view_from_mask(mask.astype(np.uint8))
        loops = spec_mod.traced_outline(view)
        if not loops:
            return []
        elements = [e for l in loops if l["type"] != "circle" for e in l["elements"]]
        elements += [dict(l, type="circle") for l in loops if l["type"] == "circle"]
        if not elements:
            return []
        drawn = [drawn_points(e) for e in elements]
        to_px, scale = transform_ideal(rec)
        ideal_px = np.concatenate([to_px(TC.polyline(e)) for loop in rec["loops"] for e in loop])
        cost, flip, s, shift = align(drawn, ideal_px)
        if cost > 25:
            return []
        moved = []
        for d in drawn:
            q = d.copy()
            q[:, 1] *= flip
            moved.append(q * s + shift)
        rows = []
        for loop in rec["loops"]:
            total = sum(float(np.sum(np.linalg.norm(np.diff(to_px(TC.polyline(e)), axis=0), axis=1)))
                        for e in loop)
            for e in loop:
                pts = to_px(TC.polyline(e))
                length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                hits = []
                for p in pts[::3]:
                    dists = [float(np.min(np.linalg.norm(mv - p, axis=1))) for mv in moved]
                    j = int(np.argmin(dists))
                    if dists[j] < 12:
                        hits.append(elements[j]["type"])
                if not hits:
                    continue
                curved = sum(1 for h in hits if h in CURVED) / len(hits)
                rows.append({"part": part, "type": e["type"], "span": e.get("span_deg"),
                             "r_px": (e.get("r") or 0) * scale, "length_px": length,
                             "share_of_loop": length / max(total, 1e-9),
                             "curved_frac": curved, "pieces": len(set(map(id, hits))),
                             "distinct": len(set(hits)), "align_cost": cost})
        return rows
    except Exception:
        return []


def main(limit=180):
    photo = json.load(open(os.path.join(ROOT, "data", "ab_arcs.json")))["shipped"]
    parts = [p for p, v in photo.items() if "n" in v and 1 - v.get("trivial", 0) >= 0.15]
    parts = [p for p in sorted(parts) if TC.IDEAL[p].get("loops")][:limit]
    with Pool(6) as pool:
        rows = [r for rs in pool.map(one, parts) for r in rs]
    json.dump(rows, open(os.path.join(ROOT, "data", "arc_survival.json"), "w"))
    curved = [r for r in rows if r["type"] in CURVED]
    straight = [r for r in rows if r["type"] == "line"]
    print("  %d elements matched over %d parts (%d curved, %d straight)\n"
          % (len(rows), len(set(r["part"] for r in rows)), len(curved), len(straight)))
    print("  reproduced as a curve:")
    print("    %-22s %6s %10s" % ("ground truth", "n", "curved"))
    for t in ("arc", "circle", "ellipse", "bsplinecurve", "line"):
        d = [r for r in rows if r["type"] == t]
        if d:
            print("    %-22s %6d %9.0f%%" % (t, len(d), 100 * np.mean([r["curved_frac"] for r in d])))

    def bucket(rows, key, edges, label):
        print("\n  by %s:" % label)
        print("    %-22s %6s %10s" % (label, "n", "curved"))
        for lo, hi in zip(edges, edges[1:] + [np.inf]):
            d = [r for r in rows if lo <= (r[key] or 0) < hi]
            if d:
                name = "%g-%g" % (lo, hi) if np.isfinite(hi) else ">%g" % lo
                print("    %-22s %6d %9.0f%%" % (name, len(d), 100 * np.mean([r["curved_frac"] for r in d])))

    arcs = [r for r in curved if r["type"] == "arc"]
    bucket(arcs, "span", [0, 15, 30, 60, 120, 240], "arc sweep, degrees")
    bucket(arcs, "r_px", [0, 10, 25, 60, 150], "arc radius, px")
    bucket(curved, "length_px", [0, 10, 25, 60, 150], "curve length on the page, px")
    bucket(curved, "share_of_loop", [0, 0.02, 0.05, 0.15, 0.4], "share of the loop's perimeter")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
