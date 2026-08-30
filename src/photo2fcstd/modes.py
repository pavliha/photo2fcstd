import os

from photo2fcstd import thresholds as th


def is_elevation(v):
    return (not v["shape"].get("roundish") and v["shape"]["rectangularity"] > th.ELEVATION_RECT
            and v["shape"]["hole_frac"] < th.ELEVATION_HOLE_FRAC and v["shape"]["ellipse_rms"] > th.ELEVATION_ELLIPSE_RMS)


def same_face(specs):
    if len(specs) < 2:
        return False
    elong = [v["elongation"] for v in specs]
    spread = lambda key: max(v["shape"][key] for v in specs) - min(v["shape"][key] for v in specs)
    return (max(elong) / min(elong) < th.SAME_FACE_ELONGATION
            or all(v["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE for v in specs)
            or (spread("rectangularity") < th.SAME_FACE_SPREAD and spread("solidity") < th.SAME_FACE_SPREAD
                and min(v["shape"]["rectangularity"] for v in specs) < th.SAME_FACE_RECT))


def select(specs, forced=None):
    holed = max(specs, key=lambda v: v["shape"]["hole_frac"])
    least_rect = min(specs, key=lambda v: v["shape"]["rectangularity"])
    if forced == "profile":
        return "profile", holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else least_rect
    if forced in ("plan", "revolve"):
        return forced, holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else specs[0]
    if forced == "stations":
        return "stations", specs[0]
    round_views = [v for v in specs if v["shape"].get("round")]
    roundish = [v for v in specs if v["shape"].get("roundish")]
    flat = same_face(specs)
    if roundish and any(is_elevation(v) for v in specs):
        return "revolve", max(roundish, key=lambda v: v["shape"]["ellipse"]["aspect"])
    if round_views and (flat or len(specs) == 1 or all(v["shape"].get("round") for v in specs)):
        return "revolve", max(round_views, key=lambda v: v["shape"]["ellipse"]["aspect"])
    if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE:
        return ("plan" if flat else "profile"), holed
    if least_rect["shape"]["rectangularity"] < th.PROFILE_RECT:
        return "profile", least_rect
    if flat and all(v["shape"]["rectangularity"] > th.PLAN_RECT for v in specs):
        return "plan", specs[0]
    return "stations", specs[0]


def station_views(specs):
    ranked = sorted(specs, key=lambda v: -v["elongation"])
    dropped = ranked[2]["source"] if len(ranked) == 3 else None
    views = dict(zip(("front", "side"), ranked[:2]))
    end_on = views["side"]["source"] if len(views) == 2 and views["side"]["elongation"] < th.END_ON_ELONGATION else None
    return views, dropped, end_on


def centred_loops(src, cx, cy):
    raw = [src["shape"]["raw"]] + src["shape"]["raw_holes"]
    return [[[x - cx, -(y - cy)] for x, y in loop] for loop in raw]


def revolve_from_elevation(src, elev, loops, R):
    ws = [st["width"] for st in elev["stations"]]
    k = R / (max(ws) / 2)
    zs = [(z - elev["z"][0]) * k for z in elev["z"]]
    profile = [[0, 0]] + [pt for i, w in enumerate(ws) for pt in ([w / 2 * k, zs[i]], [w / 2 * k, zs[i + 1]])] + [[0, zs[-1]]]
    return {"source": src["source"], "profile": profile, "holes": [l for l in loops[1:] if l["type"] == "circle"],
            "R": R, "rings": [], "generic": True,
            "note": "revolve of the elevation's half profile (%s), radius scaled to the round view" % os.path.basename(elev["source"])}


def revolve_from_prior(src, loops, R, rings, thickness_px, rim_px):
    total = th.REVOLVE_FRAC * 2 * R
    walls = [R - h["r"] for h in loops[1:] if h["type"] == "circle" and h["r"] > th.RING_HOLE_FRACTION_OF_R * R]
    if walls:
        total = min(total, 2 * min(walls))
    t = thickness_px if thickness_px is not None else 0.5 * total
    h = rim_px if rim_px is not None else total
    profile = ([[0, 0], [R, 0], [R, h], [rings[0] * R, h], [rings[0] * R, t], [0, t]] if rings
               else [[0, 0], [R, 0], [R, t], [0, t]])
    return {"source": src["source"], "profile": profile, "holes": [l for l in loops[1:] if l["type"] == "circle"],
            "R": R, "rings": rings, "t": t, "h": h,
            "note": "revolve of a half profile: R and rim radius from the photo; floor thickness and rim height are NOT visible from above - measure them"}


def outline_depth(src, others, mode, thickness_px):
    if mode == "plan":
        if thickness_px is not None:
            return thickness_px, "plate thickness from --thickness-px (px units)"
        return (th.PLATE_FRAC * src["length_px"],
                "plate thickness: NOT visible in plan photos, guessed as %.0f%% of length - set it from a caliper (px units)" % (th.PLATE_FRAC * 100))
    extent = max(src["shape"]["bbox"])
    other = max(others, key=lambda v: v["shape"]["rectangularity"]) if others else None
    if other is not None and other["shape"]["rectangularity"] > th.PLAN_RECT:
        width = max(st["width"] for st in other["stations"])
        return (width * extent / other["length_px"],
                "extrusion length = width of %s scaled by the profile extent (px units)" % os.path.basename(other["source"]))
    return (min(src["shape"]["stroke_px"], th.PLATE_FRAC * src["length_px"]),
            "no plain elevation photo: depth guessed as min(section stroke, %.0f%% of length) - measure it (px units)" % (th.PLATE_FRAC * 100))
