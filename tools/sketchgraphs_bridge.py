"""SketchGraphs sketches, rendered and encoded the way `sketchnet` expects.

`sketchnet` learns the training parts almost perfectly - 96% correct primitive count - and takes 40%
across a part split, on 967 parts. The formulation is not the problem, the sample is: PICASSO used
1.53 million sketches. SketchGraphs has 15 million real CAD sketches whose entities are exactly the
primitives we predict, so the shortage is fixable.
"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SP = os.environ.get("P2F_SG", "/private/tmp/claude-501/-Users-pavliha-Code-photo2fcstd/"
                              "645d85db-e384-4236-9a84-ba2c794f26e8/scratchpad")
sys.path.insert(0, os.path.join(SP, "sg"))
from photo2fcstd import sketchnet as SN  # noqa: E402

CANVAS = 768


def primitives_of(sketch):
    """A SketchGraphs sketch as our line / arc / circle, or None if it uses something else."""
    out = []
    for e in sketch.entities.values():
        if getattr(e, "isConstruction", False):
            continue
        kind = type(e).__name__
        if kind == "Point":
            continue
        if kind == "Line":
            a = np.array([e.pntX + e.dirX * e.startParam, e.pntY + e.dirY * e.startParam])
            b = np.array([e.pntX + e.dirX * e.endParam, e.pntY + e.dirY * e.endParam])
            out.append({"type": "line", "p0": a.tolist(), "p1": b.tolist()})
        elif kind == "Circle":
            out.append({"type": "circle", "cx": float(e.xCenter), "cy": float(e.yCenter),
                        "r": float(abs(e.radius))})
        elif kind == "Arc":
            try:
                a = np.asarray(e.start_point, float)
                b = np.asarray(e.end_point, float)
            except Exception:
                return None
            c = np.array([e.xCenter, e.yCenter], float)
            va, vb = a - c, b - c
            out.append({"type": "arc", "p0": a.tolist(), "p1": b.tolist(),
                        "cx": float(c[0]), "cy": float(c[1]), "r": float(abs(e.radius)),
                        "ccw": bool(va[0] * vb[1] - va[1] * vb[0] > 0)})
        else:
            return None
    return out or None


def to_canvas(els, n=CANVAS, margin=0.08):
    """Scale a sketch into the same box `synth` renders PrintCAD sketches into."""
    pts = []
    for e in els:
        if e["type"] == "circle":
            pts += [[e["cx"] - e["r"], e["cy"] - e["r"]], [e["cx"] + e["r"], e["cy"] + e["r"]]]
        else:
            pts += [e["p0"], e["p1"]]
            if e["type"] == "arc":
                pts += [[e["cx"] - e["r"], e["cy"] - e["r"]], [e["cx"] + e["r"], e["cy"] + e["r"]]]
    p = np.asarray(pts, float)
    lo, hi = p.min(axis=0), p.max(axis=0)
    size = float(np.max(hi - lo))
    if not np.isfinite(size) or size <= 0:
        return None
    k = (1 - 2 * margin) * n / size
    off = n / 2 - (lo + hi) / 2 * k
    out = []
    for e in els:
        f = dict(e)
        if e["type"] == "circle":
            f["cx"], f["cy"] = float(e["cx"] * k + off[0]), float(e["cy"] * k + off[1])
            f["r"] = float(e["r"] * k)
        else:
            f["p0"] = (np.asarray(e["p0"]) * k + off).tolist()
            f["p1"] = (np.asarray(e["p1"]) * k + off).tolist()
            if e["type"] == "arc":
                f["cx"], f["cy"] = float(e["cx"] * k + off[0]), float(e["cy"] * k + off[1])
                f["r"] = float(e["r"] * k)
        out.append(f)
    return out


def render(els, n=CANVAS, thickness=3):
    """The sketch as strokes, which is what a CAD sketch looks like - not a filled silhouette."""
    import cv2
    img = np.zeros((n, n), np.uint8)
    for e in els:
        if e["type"] == "circle":
            cv2.circle(img, (int(round(e["cx"])), int(round(e["cy"]))),
                       max(int(round(e["r"])), 1), 255, thickness, cv2.LINE_AA)
        elif e["type"] == "arc":
            a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
            a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
            lo, hi = (a0, a1) if e["ccw"] else (a1, a0)
            cv2.ellipse(img, (int(round(e["cx"])), int(round(e["cy"]))),
                        (max(int(round(e["r"])), 1),) * 2, 0, lo, hi if hi > lo else hi + 360,
                        255, thickness, cv2.LINE_AA)
        else:
            cv2.line(img, tuple(np.round(e["p0"]).astype(int)), tuple(np.round(e["p1"]).astype(int)),
                     255, thickness, cv2.LINE_AA)
    return img


def convert(sketch):
    """One SketchGraphs sketch to (image, encoded rows), or None if unusable."""
    import cv2
    els = primitives_of(sketch)
    if not els or len(els) > SN.SLOTS:
        return None
    els = to_canvas(els)
    if els is None:
        return None
    img = render(els)
    if img.sum() < 255 * 200:
        return None
    small = cv2.resize(img.astype(np.float32) / 255.0, (SN.SIDE, SN.SIDE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32), SN.encode(els, (np.zeros(2), float(CANVAS))).astype(np.float32)
