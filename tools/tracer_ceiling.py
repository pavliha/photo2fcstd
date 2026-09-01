"""Is the missing curve the tracer's fault or the photograph's?

The whole remaining structural gap is arcs: straight elements come out at 6.22 against a real 6.78,
curved ones at 1.61 against 5.17. Loosening the gates does not fix it (the looser gate fires on 54%
of parts that have no curve at all), and two learned attempts lost. Before anything else is tried,
this settles whether the ceiling is capture or code, by handing the tracer a perfect drawing of the
true face - rasterised from the ideal sketch with its arcs actually sampled as arcs, square on, no
walls, no tilt, no matting - and asking how many curves it gets back.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import sketch_score as SS  # noqa: E402

SIZE = 900
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")


def polyline(element):
    """Every ideal element already carries `xy`, a dense sample in the sketch plane.

    Rebuilding it from centres and endpoints drops bsplinecurve and ellipse, which have neither,
    and those are 338 of the 2,163 elements - the complex parts, which is exactly the population
    this measurement is about.
    """
    return np.array(element["xy"], float)


def rasterise(record):
    """The ideal face, drawn exactly, filling the frame."""
    loops = []
    for loop in record["loops"]:
        pts = np.concatenate([polyline(e) for e in loop])
        loops.append(pts)
    allp = np.concatenate(loops)
    lo, hi = allp.min(0), allp.max(0)
    span = max(hi - lo)
    if span <= 0:
        return None
    scale = (SIZE * 0.8) / span
    img = np.zeros((SIZE, SIZE), np.uint8)
    order = sorted(range(len(loops)), key=lambda i: -cv2.contourArea(
        np.round((loops[i] - lo) * scale).astype(np.int32)))
    for rank, i in enumerate(order):
        q = np.round((loops[i] - lo) * scale + SIZE * 0.1).astype(np.int32)
        cv2.fillPoly(img, [q], 0 if rank else 1)
    return img > 0


def one(part):
    from photo2fcstd import analysis, spec as spec_mod
    rec = IDEAL[part]
    try:
        mask = rasterise(rec)
        if mask is None or mask.sum() < 500:
            return part, None
        view = analysis.view_from_mask(mask.astype(np.uint8))
        loops = spec_mod.traced_outline(view)
        if not loops:
            return part, None
        s = SS.score_one({"outline": {"loops": loops}}, rec)
        v = SS.verdict(s)
        m, i = s["counts_mine"], s["counts_ideal"]
        v["curve_mine"] = sum(c for k, c in m.items() if k in CURVES)
        v["curve_ideal"] = sum(c for k, c in i.items() if k in CURVES)
        v["n_mine"], v["n_ideal"] = sum(m.values()), sum(i.values())
        return part, v
    except Exception as e:
        return part, {"error": str(e)[:60]}


IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def main(limit=250):
    """Compare on the parts the photo pipeline was measured on, or the populations differ."""
    photo = json.load(open(os.path.join(ROOT, "data", "ab_arcs.json")))["shipped"]
    parts = [p for p, v in photo.items() if "n" in v and 1 - v.get("trivial", 0) >= 0.15]
    parts = [p for p in sorted(parts) if IDEAL[p].get("loops")][:limit]
    with Pool(6) as pool:
        rows = dict(pool.map(one, parts))
    ok = {p: v for p, v in rows.items() if v and "region_iou" in v}
    lost = [p for p, v in rows.items() if not v or "region_iou" not in v]
    json.dump(ok, open(os.path.join(ROOT, "data", "tracer_ceiling.json"), "w"))
    V = list(ok.values())
    ph = {p: photo[p] for p in ok}
    print("  same parts both columns: %d of %d rasterised, %d could not be drawn\n" % (len(V), len(rows), len(lost)))
    print("  %-26s %8s %8s" % ("", "perfect", "photos"))
    print("  %-26s %8.2f %8.2f" % ("curved primitives drawn", np.mean([v["curve_mine"] for v in V]), np.mean([x["curve_mine"] for x in ph.values()])))
    print("  %-26s %8.2f %8.2f" % ("really there", np.mean([v["curve_ideal"] for v in V]), np.mean([x["curve_ideal"] for x in ph.values()])))
    print("  %-26s %8.2f %8.2f" % ("elements drawn", np.mean([v["n_mine"] for v in V]), np.mean([x["n"] for x in ph.values()])))
    print("  %-26s %8.2f %8.2f" % ("really there", np.mean([v["n_ideal"] for v in V]), np.mean([sum(IDEAL[p].get("counts", {}).values()) for p in ok])))
    print()
    print("  %-26s %8.3f %8.3f" % ("region IoU", np.mean([v["region_iou"] for v in V]), np.mean([x["iou"] for x in ph.values()])))
    print("  %-26s %8.3f %8s" % ("structure", np.mean([v["structure"] for v in V]), "-"))
    print("  %-26s %8.3f %8s" % ("  curves term", np.mean([v["curves"] for v in V]), "-"))
    print("  %-26s %8.3f %8s" % ("  elements term", np.mean([v["elements"] for v in V]), "-"))
    print("  %-26s %7.0f%% %7.0f%%" % ("exact primitives", 100*np.mean([v["exact"] for v in V]),
                                       100*np.mean([x["exact"] for x in ph.values()])))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
