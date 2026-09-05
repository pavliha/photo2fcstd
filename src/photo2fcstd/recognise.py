"""Recognition-driven CAD: a vision model states the structure, geometry measures it.

The contract is a small JSON a VLM produces from the photos. It never gives coordinates - only
identity and structure (which photo is face-on, the outline kind, what openings and fastener
patterns exist). The geometry engine then measures every dimension from the recognised face photo,
so precision stays with the tracer and the model supplies only the prior no silhouette carries.

    rec = {
      "face_photo_index": 2,            # which of the given photos is square-on to the main face
      "outline": "square"|"rect"|"disc"|"trace",  # 'trace' = let the tracer draw a freeform outline
      "openings": ["bore"|"none"],      # dark interior openings to recover and fit as circles
      "screws": 0-8,                    # corner/edge fastener holes to place at plate corners
      "depth_ratio": 0.0-2.0            # depth / longest-face-dimension, read from a side photo
    }

`recognise_prompt()` returns the exact instruction to hand a VLM with the photos; `build(rec, ...)`
turns its answer into a buildable spec. Wiring a live model is one call - see cli use.
"""
import os

import numpy as np


def recognise_prompt(n_photos):
    return (
        "You are given %d photographs of a single manufactured part. Return ONLY a JSON object with "
        "these keys, describing the part's primary extruded face - not coordinates, only structure:\n"
        '  "face_photo_index": integer 0..%d - the photo looking square-on to the largest flat face\n'
        '  "outline": one of "square","rect","disc","trace" - the face boundary shape; use "trace" '
        "if it is an irregular profile the tracer should draw freehand\n"
        '  "openings": list of "bore" (a large central round opening) or [] if none\n'
        '  "screws": integer 0..8 - visible corner/edge fastener holes\n'
        '  "depth_ratio": number - the part depth divided by the longest face dimension, judged '
        "from a side view if present, else your best estimate\n"
        "Judge only what is visible; do not invent features." % (n_photos, n_photos - 1))


VISION_CMD = os.environ.get("P2F_VISION_CMD", "claude")


def recognise_live(photos, model=None):
    """Ask the installed vision model to fill the contract from the photos. Returns the rec dict.

    Shells out to the Claude Code CLI in print mode - a real, authenticated vision call, no key
    plumbing. Set P2F_VISION_CMD to point at any CLI that takes a prompt and reads image paths.
    Images are copied to a temp dir the tool is allowed to read.
    """
    import json as _json
    import shutil
    import subprocess
    import tempfile
    d = tempfile.mkdtemp(prefix="recog_")
    local = []
    for i, p in enumerate(photos):
        q = os.path.join(d, "img%02d%s" % (i, os.path.splitext(p)[1] or ".jpg"))
        shutil.copy(p, q)
        local.append(q)
    listing = "\n".join("  photo %d: %s" % (i, q) for i, q in enumerate(local))
    prompt = ("Read these %d photographs of one manufactured part:\n%s\n\n%s\n"
              "Reply with ONLY the JSON object, no prose, no code fence."
              % (len(local), listing, recognise_prompt(len(local))))
    cmd = [VISION_CMD, "-p", prompt, "--allowedTools", "Read"]
    if model:
        cmd += ["--model", model]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    shutil.rmtree(d, ignore_errors=True)
    start = out.find("{")
    end = out.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("vision model returned no JSON:\n" + out[-400:])
    return _json.loads(out[start:end + 1])


def build(rec, photos, name="part", length_mm=None):
    from photo2fcstd import cli
    from photo2fcstd.trace import fit_ellipse, outline, segment_photo, upright_mask
    os.environ["P2F_RECOVER_DARK"] = "1" if rec.get("openings") else os.environ.get("P2F_RECOVER_DARK", "0")
    face = photos[int(rec.get("face_photo_index", 0))]
    mask, _ = upright_mask(segment_photo(face))
    poly, sh = outline(mask)
    W, H = float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))
    cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
    kind = rec.get("outline", "trace")
    loops = []
    if kind in ("square", "rect"):
        loops.append(_rect_loop(cx, cy, W, H, square=(kind == "square")))
    elif kind == "disc":
        loops.append({"type": "circle", "cx": cx, "cy": cy, "r": min(W, H) / 2})
    else:
        from photo2fcstd import modes, spec as spec_mod
        traced = spec_mod.traced_outline(_view_of(face))
        loops.extend(traced or [_rect_loop(cx, cy, W, H, square=False)])
    if "bore" in (rec.get("openings") or []) and sh["holes"]:
        f = fit_ellipse(np.asarray(max(sh["holes"], key=len), float))
        loops.append({"type": "circle", "cx": float(f["cx"]), "cy": float(f["cy"]), "r": float(f["a"])})
    for sx, sy in _corners(cx, cy, W, H)[:int(rec.get("screws", 0))]:
        loops.append({"type": "circle", "cx": sx, "cy": sy, "r": 0.03 * min(W, H)})
    depth_px = float(rec.get("depth_ratio", 0.5)) * max(W, H)
    spec = {"name": name, "mode": "recognised", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: set from one caliper reading; the sheet is in pixels until you set scale",
            "views": {}, "revolve": None, "stl": None, "measured": [],
            "outline": {"source": face, "loops": loops, "depth_px": depth_px,
                        "depth_note": "depth from the recognised side proportion (px units)",
                        "depth_trusted": False}}
    return spec


def _view_of(path):
    from photo2fcstd import analysis
    return analysis.view(path)


def _rect_loop(cx, cy, w, h, square):
    a, b = (min(w, h) / 2, min(w, h) / 2) if square else (w / 2, h / 2)
    c = [(-a, -b), (a, -b), (a, b), (-a, b)]
    pts = [(cx + x, cy + y) for x, y in c]
    els = [{"type": "line", "p0": list(pts[i]), "p1": list(pts[(i + 1) % 4])} for i in range(4)]
    return {"type": "loop", "elements": els, "kinds": ["H", "V", "H", "V"], "joins": [""] * 4}


def _corners(cx, cy, w, h, inset=0.08):
    a = min(w, h) / 2 * (1 - inset)
    return [(cx - a, cy - a), (cx + a, cy - a), (cx + a, cy + a), (cx - a, cy + a)]
