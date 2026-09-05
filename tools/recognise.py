"""LLM-recognised sketch: identity and structure from vision, geometry for precision.

The sixteen learned replacements all fed a contour to a small model and lost. This is the other
shape - the one that wins here: a vision model states WHAT the part is (object class, which photo is
face-on, what primitives its sketch holds, which regions are openings), and the geometric engine
measures WHERE (exact positions and radii from the chosen photo's silhouette). Recognition supplies
the prior; the tracer supplies the precision. The recognition below is a structured description a
VLM produces from the photos; here it is written out for the fan as a worked example.
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def build_from_recognition(rec, out_spec, out_fcstd, length_mm=None):
    """rec: {face_photo, plate: 'square'|'rect', bore: bool, screws: int, depth_ratio}.

    Geometry measures the plate outline and the bore radius from the recognised face photo; the
    recogniser only says they exist and which photo to read. Depth from the recognised side ratio.
    """
    from photo2fcstd import analysis, cli
    from photo2fcstd.trace import fit_ellipse, outline, segment_photo, upright_mask
    os.environ["P2F_RECOVER_DARK"] = "1"
    mask, _ = upright_mask(segment_photo(rec["face_photo"]))
    poly, sh = outline(mask)
    W = float(np.ptp(poly[:, 0])); H = float(np.ptp(poly[:, 1]))
    cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
    loops = [{"type": "loop",
              "elements": _square(cx, cy, W, H),
              "kinds": ["H", "V", "H", "V"], "joins": ["", "", "", ""]}]
    if rec.get("bore") and sh["holes"]:
        h = np.asarray(max(sh["holes"], key=len), float)
        f = fit_ellipse(h)
        loops.append({"type": "circle", "cx": float(f["cx"]), "cy": float(f["cy"]), "r": float(f["a"])})
    for j, (sx, sy) in enumerate(_corners(cx, cy, W, H, inset=0.08)[:rec.get("screws", 0)]):
        r = 0.03 * min(W, H)
        loops.append({"type": "circle", "cx": sx, "cy": sy, "r": r})
    depth_px = rec.get("depth_ratio", 0.5) * max(W, H)
    spec = {"name": rec.get("name", "part"), "mode": "recognised", "mm_per_px": 1.0,
            "unit": "px", "scale_note": "UNSCALED: one caliper reading; the sheet is in pixels until you set scale",
            "views": {}, "revolve": None, "stl": None, "measured": [],
            "outline": {"source": rec["face_photo"], "loops": loops, "depth_px": depth_px,
                        "depth_note": "depth from the recognised side-view proportion (px units)",
                        "depth_trusted": False}}
    json.dump(spec, open(out_spec, "w"))
    return cli.freecad_build(out_spec, out_fcstd)


def _square(cx, cy, w, h):
    a = min(w, h) / 2
    c = [(-a, -a), (a, -a), (a, a), (-a, a)]
    pts = [(cx + x, cy + y) for x, y in c]
    return [{"type": "line", "p0": list(pts[i]), "p1": list(pts[(i + 1) % 4])} for i in range(4)]


def _corners(cx, cy, w, h, inset):
    a = min(w, h) / 2 * (1 - inset)
    return [(cx - a, cy - a), (cx + a, cy - a), (cx + a, cy + a), (cx - a, cy + a)]


if __name__ == "__main__":
    U = "/Users/pavliha/.claude/uploads/645d85db-e384-4236-9a84-ba2c794f26e8/"
    rec = {"name": "fan", "face_photo": U + "3ab300ee-image.jpg",
           "plate": "square", "bore": True, "screws": 4, "depth_ratio": 0.42}
    r = build_from_recognition(rec, os.path.join(ROOT, "runs/fan/recognised.spec.json"),
                               os.path.join(ROOT, "runs/fan/recognised.FCStd"))
    print("BUILD valid=%s solids=%s loops=%s unsolved=%d"
          % (r["valid"], r["solids"], sum(1 for s in r["sketches"].values()),
             sum(1 for s in r["sketches"].values() if s["solve"] != 0)))
