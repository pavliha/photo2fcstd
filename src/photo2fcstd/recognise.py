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
        '  "grille": {"rings": int, "spokes": int} or null - concentric grille rings and straight '
        "spoke bars across the centre (an X is 2 spokes); null if the opening is plain\n"
        '  "single_extrusion": true|false - true if the whole part is one flat plate/profile of '
        "constant thickness (route to the measured pipeline); false if it has a box body, a "
        "recessed face, stacked levels or any depth structure (route to the feature builder)\n"
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
    from photo2fcstd import cli, trace
    from photo2fcstd.trace import fit_ellipse, outline, segment_photo, upright_mask
    want_bore = bool(rec.get("openings"))
    if want_bore:
        trace.RECOVER_DARK = True   # module flag, not just env - trace read env at import
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
        bx, by, br = float(f["cx"]), float(f["cy"]), float(f["a"])
        g = rec.get("grille")
        if g and int(g.get("rings", 0)) > 1:
            nr = int(g["rings"])
            for k in range(nr):
                rk = br * (1.0 - 0.82 * k / max(nr - 1, 1))
                loops.append({"type": "circle", "cx": bx, "cy": by, "r": rk})
            bw = 0.04 * br
            for a in _spoke_angles(int(g.get("spokes", 0))):
                u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]])
                c = np.array([bx, by])
                pts = [c - u*br - v*bw, c + u*br - v*bw, c + u*br + v*bw, c - u*br + v*bw]
                loops.append({"type": "loop",
                              "elements": [{"type": "line", "p0": pts[i].tolist(),
                                            "p1": pts[(i+1)%4].tolist()} for i in range(4)],
                              "kinds": ["F","F","F","F"], "joins": ["","","",""]})
        else:
            loops.append({"type": "circle", "cx": bx, "cy": by, "r": br})
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


def grille_loops(cx, cy, R, n_rings, n_spokes, web=0.10):
    """A fan grille as one valid region: outer disc minus the open annular-sectors.

    The material is `n_rings` concentric bands plus `n_spokes` radial arms, all of relative width
    `web`; everything between is an opening. Returns [outer circle] + [sector-opening hole loops],
    which padded gives the connected grille web - a manufacturable feature, not overlapping circles.
    """
    loops = [{"type": "circle", "cx": cx, "cy": cy, "r": R}]
    t = web * R                       # material half-pitch
    edges = np.linspace(R, t, n_rings + 1)   # ring band boundaries, outer -> inner
    sw = web * np.pi                  # spoke half-angle
    arms = [i * 2 * np.pi / max(n_spokes * 2, 1) for i in range(n_spokes * 2)] if n_spokes else [0.0]
    for k in range(n_rings):
        r_out, r_in = edges[k] - t / 2, edges[k + 1] + t / 2
        if r_out - r_in < t:
            continue
        for a in range(len(arms)):
            a0 = arms[a] + sw
            a1 = arms[(a + 1) % len(arms)] - sw
            span = (a1 - a0) % (2 * np.pi)
            if not (0.02 < span < 2 * np.pi):
                continue
            loops.append(_sector(cx, cy, r_in, r_out, a0, a0 + span))
    return loops


def _sector(cx, cy, r_in, r_out, a0, a1):
    p = lambda r, a: [cx + r * np.cos(a), cy + r * np.sin(a)]
    return {"type": "loop", "elements": [
        {"type": "arc", "p0": p(r_out, a0), "p1": p(r_out, a1), "cx": cx, "cy": cy, "r": r_out, "ccw": True},
        {"type": "line", "p0": p(r_out, a1), "p1": p(r_in, a1)},
        {"type": "arc", "p0": p(r_in, a1), "p1": p(r_in, a0), "cx": cx, "cy": cy, "r": r_in, "ccw": False},
        {"type": "line", "p0": p(r_in, a0), "p1": p(r_out, a0)}],
        "kinds": ["A", "F", "A", "F"], "joins": ["", "", "", ""]}


def program_prompt(n):
    return (
        "You are given %d photographs of one manufactured part. Classify it - a JSON object, "
        "structure only, no coordinates:\n"
        '  "face_photo_index": int 0..%d - the photo square-on to the largest flat face\n'
        '  "revolve": true|false - true if the whole part is turned round a central axis '
        "(a bottle, knob, wheel, cup); then features are ignored and the side profile is revolved\n"
        '  "part_class": a standard class name if you recognise one ("fan_guard","bottle",'
        '"bracket","plate","enclosure"), else null\n'
        '  "single_extrusion": true|false - one flat profile of constant thickness\n'
        '  "openings": ["bore"] if there is a large central round opening, else []\n'
        '  "grille": {"rings": int, "spokes": int} or null\n'
        "Judge only what is visible.\n"
        "Reply with ONLY the JSON." % (n, n - 1))


def revolve_spec(photos, rec, name="part", samples=48):
    from photo2fcstd.trace import segment_photo, upright_mask
    face = photos[int(rec.get("face_photo_index", 0))]
    mask, _ = upright_mask(segment_photo(face))
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    axis = (xs.min() + xs.max()) / 2.0
    hs = np.linspace(y0, y1, samples)
    prof = [[0.0, 0.0]]
    for y in hs:
        row = xs[np.abs(ys - y) <= max((y1 - y0) / samples, 1)]
        if len(row) == 0:
            continue
        r = max(float(row.max() - axis), float(axis - row.min()))
        prof.append([round(r, 2), round(float(y - y0), 2)])
    prof.append([0.0, round(float(y1 - y0), 2)])
    return {"name": name, "mode": "revolve", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: set from one caliper reading",
            "views": {}, "outline": None, "stl": None, "measured": [],
            "revolve": {"generic": True, "profile": straight_or_traced(prof), "holes": [], "rings": []}}


def straight_or_traced(prof, spread_max=1.25):
    mid = np.array(prof[1:-1], float)
    mid = mid[int(0.1 * len(mid)):max(int(0.9 * len(mid)), 1)]
    if len(mid) < 4 or np.percentile(mid[:, 0], 90) / max(np.percentile(mid[:, 0], 10), 1e-6) > spread_max:
        return prof
    r, h = round(float(np.median(mid[:, 0])), 2), prof[-1][1]
    return [[0.0, 0.0], [r, 0.0], [r, h], [0.0, h]]


def _to_jpeg(src, dst):
    ext = os.path.splitext(src)[1].lower()
    if ext in (".jpg", ".jpeg", ".png"):
        import shutil
        shutil.copy(src, dst)
        return dst
    from PIL import Image
    from photo2fcstd.trace import load
    arr = load(src)
    Image.fromarray((arr * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")).save(dst, "JPEG")
    return dst


def _recognition_cache_path(photos):
    import hashlib
    h = hashlib.sha1()
    for p in photos:
        st = os.stat(p)
        h.update(("%s|%d|%d" % (os.path.abspath(p), st.st_size, int(st.st_mtime))).encode())
    h.update(program_prompt(len(photos)).encode())
    d = os.path.join(os.path.expanduser("~/.cache/photo2fcstd"), "recognition")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h.hexdigest() + ".json")


def _recognise_program_live(photos):
    import json as _json, shutil, subprocess, tempfile
    cache = _recognition_cache_path(photos)
    if os.path.exists(cache):
        return _json.load(open(cache))
    d = tempfile.mkdtemp(prefix="prog_")
    local = [_to_jpeg(p, os.path.join(d, "img%02d.jpg" % i)) for i, p in enumerate(photos)]
    listing = "\n".join("  photo %d: %s" % (i, q) for i, q in enumerate(local))
    prompt = "Read these %d photos:\n%s\n\n%s" % (len(local), listing, program_prompt(len(local)))
    out = subprocess.run([VISION_CMD, "-p", prompt, "--allowedTools", "Read"],
                         capture_output=True, text=True, timeout=300).stdout
    shutil.rmtree(d, ignore_errors=True)
    s, e = out.find("{"), out.rfind("}")
    if s < 0:
        raise ValueError("no JSON:\n" + out[-400:])
    rec = _json.loads(out[s:e + 1])
    _json.dump(rec, open(cache, "w"))
    return rec


def build_program(recs, photos, name="part", face_index=0):
    """A multi-feature part from recognition: a box body, a bore, corner holes - each a real feature.

    `recs` is the structured recognition; the face photo gives the plate outline (measured) and the
    bore radius. Produces a features spec the build engine turns into pad + pocket features.
    """
    from photo2fcstd import trace
    from photo2fcstd.trace import fit_ellipse, outline, segment_photo, upright_mask
    if recs.get("openings"):
        trace.RECOVER_DARK = True
    face = photos[int(recs.get("face_photo_index", face_index))]
    mask, _ = upright_mask(segment_photo(face))
    poly, sh = outline(mask)
    W, H = float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))
    cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
    depth = float(recs.get("depth_ratio", 0.5)) * max(W, H)
    plate = _rect_loop(cx, cy, W, H, square=(recs.get("outline") == "square"))
    feats = [{"op": "pad", "depth_px": depth, "loops": [plate]}]
    if "bore" in (recs.get("openings") or []) and sh["holes"]:
        f = fit_ellipse(np.asarray(max(sh["holes"], key=len), float))
        feats.append({"op": "pocket", "through": True,
                      "loops": [{"type": "circle", "cx": float(f["cx"]), "cy": float(f["cy"]), "r": float(f["a"])}]})
    g = recs.get("grille")
    if g and int(g.get("rings", 0)) >= 2 and sh["holes"]:
        f0 = fit_ellipse(np.asarray(max(sh["holes"], key=len), float))
        gl = grille_loops(float(f0["cx"]), float(f0["cy"]), float(f0["a"]) * 1.08,
                          int(g["rings"]), int(g.get("spokes", 0)), web=0.035)
        feats.append({"op": "pad", "plane_mm": depth * 0.94, "depth_px": 0.10 * depth, "loops": gl})
    screws = int(recs.get("screws", 0))
    if screws:
        holes = [{"type": "circle", "cx": sx, "cy": sy, "r": 0.03 * min(W, H)}
                 for sx, sy in _corners(cx, cy, W, H)[:screws]]
        feats.append({"op": "pocket", "through": True, "loops": holes})
    return {"name": name, "mode": "features", "mm_per_px": 1.0, "unit": "px",
            "scale_note": "UNSCALED: set from one caliper reading; the sheet is in pixels until you set scale",
            "views": {}, "outline": None, "revolve": None, "stl": None, "measured": [],
            "features": feats}


def route(photos, name="part", rec=None):
    from photo2fcstd import analysis, spec as spec_mod
    rec = rec if rec is not None else recognise_live(photos)
    if rec.get("single_extrusion"):
        return spec_mod.assemble([analysis.view(p) for p in photos], name=name), rec
    return build_program(rec, photos, name=name), rec


def count_rings(gray, cx, cy, r_bore):
    from scipy.signal import find_peaks
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    counts = []
    for a in angles:
        rs = np.linspace(0.08 * r_bore, 0.95 * r_bore, 120)
        xs = np.clip((cx + rs * np.cos(a)).astype(int), 0, gray.shape[1] - 1)
        ys = np.clip((cy + rs * np.sin(a)).astype(int), 0, gray.shape[0] - 1)
        prof = gray[ys, xs].astype(float)
        prof = (prof - prof.min()) / (np.ptp(prof) + 1e-6)
        peaks, _ = find_peaks(prof, prominence=0.25, distance=4)
        counts.append(len(peaks))
    return int(np.median(counts)) if counts else 0


def _circle_fit(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    return cx, cy, float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def fit_fan_guard(face, frame_w_mm=80.0, rec=None):
    import cv2
    from scipy import ndimage
    from photo2fcstd import trace
    from photo2fcstd.trace import fit_ellipse, load, outline, segment_photo, upright_mask
    trace.RECOVER_DARK = True
    mask, angle = upright_mask(segment_photo(face))
    poly, sh = outline(mask)
    ys, xs = np.nonzero(mask)
    frame_px = float(max(np.ptp(xs), np.ptp(ys)))
    s = frame_w_mm / frame_px
    ledger = {"frame_w": "required"}
    p = {"frame_w": frame_w_mm}

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    rs = []
    P4 = np.asarray(poly, float)
    for i, c in enumerate(P4):
        near = cnt[np.linalg.norm(cnt - c, axis=1) < 0.12 * frame_px]
        for nb in (P4[i - 1], P4[(i + 1) % len(P4)]):        # drop points on the straight edges
            e = (nb - c) / (np.linalg.norm(nb - c) + 1e-9)
            off = np.abs((near - c) @ np.array([-e[1], e[0]]))
            near = near[off > 0.01 * frame_px]
        if len(near) >= 8:
            _, _, r = _circle_fit(near)
            if 0.005 * frame_px < r < 0.3 * frame_px:
                rs.append(r)
    if len(rs) >= 3:
        p["corner_r"], ledger["corner_r"] = round(float(np.median(rs)) * s, 2), "measured"
    else:
        p["corner_r"], ledger["corner_r"] = round(0.05 * frame_w_mm, 2), "default"

    hole = max(sh["holes"], key=len) if sh["holes"] else None
    if hole is not None:
        f = fit_ellipse(np.asarray(hole, float))
        p["bore_d"], ledger["bore_d"] = round(2 * float(f["a"]) * s, 2), "measured"
        img = load(face)
        gray = ndimage.rotate(img.mean(axis=2) if img.ndim == 3 else img.astype(float), -angle, reshape=True, order=1)
        n = count_rings(gray, float(f["cx"]), float(f["cy"]), float(f["a"]))
        g = (rec or {}).get("grille") or {}
        p["rings"], ledger["rings"] = (n, "measured") if n >= 2 else (int(g.get("rings", 4)), "default")
    else:
        p["bore_d"], ledger["bore_d"] = round(0.9 * frame_w_mm, 2), "default"
        p["rings"], ledger["rings"] = 4, "default"

    rgb = load(face)
    rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    rgb = ndimage.rotate(rgb, -angle, reshape=True, order=1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    screw = (hsv[..., 1] < 70) & (hsv[..., 2] > 120) & mask[:hsv.shape[0], :hsv.shape[1]]
    cx, cy = xs.mean(), ys.mean()
    centres = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted([cx + sx * 0.30 * frame_px, cx + sx * 0.49 * frame_px])
            y0, y1 = sorted([cy + sy * 0.30 * frame_px, cy + sy * 0.49 * frame_px])
            win = np.zeros_like(screw); win[int(y0):int(y1), int(x0):int(x1)] = True
            lab, k = ndimage.label(screw & win)
            if k:
                sizes = ndimage.sum(screw & win, lab, range(1, k + 1))
                i = int(np.argmax(sizes)) + 1
                if sizes[i - 1] > 0.0002 * frame_px ** 2:
                    centres.append(ndimage.center_of_mass(lab == i))
    if len(centres) == 4:
        c = np.array(centres)[:, ::-1]
        pitch = np.median([np.ptp(c[:, 0]), np.ptp(c[:, 1])])
        p["mount_pitch"], ledger["mount_pitch"] = round(float(pitch) * s, 2), "measured"
    else:
        p["mount_pitch"], ledger["mount_pitch"] = round(0.894 * frame_w_mm, 2), "default"
    p["mount_d"], ledger["mount_d"] = round(0.056 * frame_w_mm, 2), "default"
    p["plate_t"], ledger["plate_t"] = round(0.05 * frame_w_mm, 2), "default"
    p["wire_w"], ledger["wire_w"] = round(0.03 * frame_w_mm, 2), "default"
    return p, ledger


def fan_guard_params(rec, photos, frame_w_mm=80.0):
    from photo2fcstd import trace
    from photo2fcstd.trace import fit_ellipse, load, outline, segment_photo, upright_mask
    if rec.get("openings"):
        trace.RECOVER_DARK = True
    face = photos[int(rec.get("face_photo_index", 0))]
    mask, angle = upright_mask(segment_photo(face))
    poly, sh = outline(mask)
    frame_px = float(max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1])))
    scale = frame_w_mm / frame_px
    g = rec.get("grille") or {}
    rings = int(g.get("rings", 4))
    bore_mm = 0.9 * frame_w_mm
    if sh["holes"]:
        from scipy import ndimage
        f = fit_ellipse(np.asarray(max(sh["holes"], key=len), float))
        bore_mm = round(2 * float(f["a"]) * scale, 1)
        img = load(face)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        gray = ndimage.rotate(gray, -angle, reshape=True, order=1)
        measured = count_rings(gray, float(f["cx"]), float(f["cy"]), float(f["a"]))
        if measured >= 2:
            rings = measured
    return {"frame_w": frame_w_mm, "corner_r": round(0.05 * frame_w_mm, 1),
            "plate_t": round(0.05 * frame_w_mm, 1), "bore_d": bore_mm,
            "mount_pitch": round(0.894 * frame_w_mm, 1),
            "mount_d": round(0.056 * frame_w_mm, 1),
            "rings": rings, "wire_w": round(0.03 * frame_w_mm, 1)}


def traced_outline_mm(face, frame_w_mm, samples=180):
    from photo2fcstd import analysis, spec as spec_mod, trace
    from photo2fcstd.trace import segment_photo, upright_mask
    was = trace.RECOVER_DARK
    trace.RECOVER_DARK = False
    try:
        view = analysis.view(face)
    finally:
        trace.RECOVER_DARK = was
    loops = spec_mod.traced_outline(view)
    if not loops or loops[0]["type"] != "loop":
        return None
    pts = []
    for e in loops[0]["elements"]:
        if e["type"] == "line":
            pts.append(e["p0"])
        else:
            a0 = np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]); a1 = np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"])
            if e.get("ccw", True) and a1 < a0: a1 += 2 * np.pi
            if not e.get("ccw", True) and a1 > a0: a1 -= 2 * np.pi
            for a in np.linspace(a0, a1, 12, endpoint=False):
                pts.append([e["cx"] + e["r"] * np.cos(a), e["cy"] + e["r"] * np.sin(a)])
    P = np.asarray(pts, float)
    P -= P.mean(0)
    s = frame_w_mm / max(np.ptp(P, 0).max(), 1e-9)
    return (P * s).round(3).tolist()


def design_fan(photos, out, rec=None, frame_w_mm=80.0, depth_json=None, traced=False):
    import json, subprocess, tempfile
    rec = rec if rec is not None else recognise_live(photos)
    fits = []
    for ph in photos:
        try:
            p, l = fit_fan_guard(ph, frame_w_mm, rec=rec)
        except Exception:
            continue
        if l.get("bore_d") == "measured":
            fits.append((p, l))
    if not fits:
        face = photos[int(rec.get("face_photo_index", 0))]
        fits = [fit_fan_guard(face, frame_w_mm, rec=rec)]
    params, ledger = dict(fits[0][0]), dict(fits[0][1])
    for k in ("corner_r", "bore_d", "mount_pitch", "rings"):
        vals = [p[k] for p, l in fits if l.get(k) == "measured"]
        if vals:
            params[k] = (int(np.median(vals)) if k == "rings" else round(float(np.median(vals)), 2))
            ledger[k] = "measured (%d views)" % len(vals)
    params["box_depth"], ledger["box_depth"] = 0.0, "default (flat guard; give a 3D depth for the enclosure)"
    if depth_json and os.path.exists(depth_json):
        d3 = json.load(open(depth_json))
        if "trace_short_over_long" in d3:      # traced top footprint: flying pixels trimmed, agrees with the photo
            thin = min(d3["trace_short_over_long"], d3["trace_height_over_long"], 1.0)
            src = "measured (3D, traced footprint)"
        else:
            thin = min(d3["short_over_long"], d3["height_over_long"], 1.0)
            src = "measured (3D, percentile extents)"
        params["box_depth"], ledger["box_depth"] = round(thin * frame_w_mm, 2), src
    if traced:
        best = photos[int(rec.get("face_photo_index", 0))]
        pts = traced_outline_mm(best, frame_w_mm)
        if pts and len(pts) >= 4:
            params["outline_pts"], ledger["outline"] = pts, "traced (photo, %d points)" % len(pts)
    params = {**params, "_ledger": ledger}
    pj = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(params, pj); pj.close()
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools", "fan_guard.py")
    freecad = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
    env = dict(os.environ, P2F_PARAMS=pj.name, P2F_OUT=out)
    r = subprocess.run([freecad, tool], env=env, capture_output=True, text=True, timeout=300)
    if "SAVED" not in r.stdout:
        raise RuntimeError((r.stdout + r.stderr)[-600:])
    return out, params


def _view_of(path):
    from photo2fcstd import analysis
    return analysis.view(path)


def _spoke_angles(n):
    import numpy as np
    return [np.pi / 4 + i * np.pi / max(n, 1) for i in range(n)]


def _rect_loop(cx, cy, w, h, square):
    a, b = (min(w, h) / 2, min(w, h) / 2) if square else (w / 2, h / 2)
    c = [(-a, -b), (a, -b), (a, b), (-a, b)]
    pts = [(cx + x, cy + y) for x, y in c]
    els = [{"type": "line", "p0": list(pts[i]), "p1": list(pts[(i + 1) % 4])} for i in range(4)]
    return {"type": "loop", "elements": els, "kinds": ["H", "V", "H", "V"], "joins": [""] * 4}


def _corners(cx, cy, w, h, inset=0.08):
    a = min(w, h) / 2 * (1 - inset)
    return [(cx - a, cy - a), (cx + a, cy - a), (cx + a, cy + a), (cx - a, cy + a)]
