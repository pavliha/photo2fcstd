"""Rendered sketches paired with the primitives they were rendered from."""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import sketchnet as SN, sketch_score as SS, synth  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
PER_PART = 6
TILT = float(os.environ.get("P2F_SKETCHNET_TILT", "0"))


def primitives_of(loops):
    """The STEP sketch as line / arc / circle, in the canvas coordinates it was rendered in."""
    out = []
    for loop in loops:
        for kind, pts in loop:
            p = np.asarray(pts, float)
            if len(p) < 2:
                continue
            k = kind.lower()
            if k == "line":
                out.append({"type": "line", "p0": p[0].tolist(), "p1": p[-1].tolist()})
            elif k == "circle" and len(p) > 4:
                c = p.mean(axis=0)
                out.append({"type": "circle", "cx": float(c[0]), "cy": float(c[1]),
                            "r": float(np.median(np.linalg.norm(p - c, axis=1)))})
            else:
                c = p.mean(axis=0)
                r = float(np.median(np.linalg.norm(p - c, axis=1)))
                a, b = p[0] - c, p[-1] - c
                out.append({"type": "arc", "p0": p[0].tolist(), "p1": p[-1].tolist(),
                            "cx": float(c[0]), "cy": float(c[1]), "r": r,
                            "ccw": bool(a[0] * b[1] - a[1] * b[0] > 0)})
    return out


def one(args):
    import cv2
    part, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(PER_PART):
        try:
            loops = [synth.loop_edges(lp) for lp in IDEAL[part]["loops"]]
            loops = [lp for lp in loops if lp]
            if not loops:
                return []
            H = synth.homography(rng, tilt_deg=TILT) if TILT > 0 else np.eye(3)
            loops = [[(t, synth.apply_h(H, p)) for t, p in lp] for lp in loops]
            loops = synth.fit_canvas(loops, synth.CANVAS)
            els = primitives_of(loops)
            if not els or len(els) > SN.SLOTS:
                continue
            # strokes, not a filled silhouette, so this matches how SketchGraphs is rendered for
            # pretraining and how a traced contour is drawn at inference
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            if os.environ.get("P2F_FILLED") == "1":
                raster = (synth.rasterise(loops, synth.CANVAS).astype(np.uint8)) * 255
            else:
                from sketchgraphs_bridge import render as stroke_render
                raster = stroke_render(els, synth.CANVAS)
            if raster.sum() < 255 * 200:
                continue
            img = cv2.resize(raster.astype(np.float32) / 255.0, (SN.SIDE, SN.SIDE),
                             interpolation=cv2.INTER_AREA)
            box = (np.zeros(2), float(synth.CANVAS))
            rows.append({"part": part, "img": img.astype(np.float32),
                         "rows": SN.encode(els, box).astype(np.float32)})
        except Exception:
            continue
    return rows


def main(limit=None):
    parts = [k for k in sorted(IDEAL) if SS.trustworthy(IDEAL[k])]
    parts = parts[:limit] if limit else parts
    with Pool(6) as pool:
        got = [r for chunk in pool.map(one, [(p, i) for i, p in enumerate(parts)]) for r in chunk]
    tag = ("filled" if os.environ.get("P2F_FILLED") == "1" else "tilt%g") % TILT if os.environ.get("P2F_FILLED") != "1" else "filled"
    np.savez_compressed(os.path.join(ROOT, "data", "sketchnet_%s.npz" % tag),
                        X=np.stack([r["img"] for r in got]),
                        Y=np.stack([r["rows"] for r in got]),
                        parts=np.array([r["part"] for r in got]))
    n = np.array([r["rows"][:, 0].sum() for r in got])
    print("%d samples from %d parts, tilt %g" % (len(got), len({r["part"] for r in got}), TILT))
    print("  primitives per sketch: median %.0f, max %.0f" % (np.median(n), n.max()))
    kinds = np.stack([r["rows"] for r in got])
    present = kinds[:, :, 0] > 0.5
    for i, t in enumerate(SN.TYPES):
        print("  %-7s %6d" % (t, int(kinds[:, :, 1 + i][present].sum())))


if __name__ == "__main__":
    main()
