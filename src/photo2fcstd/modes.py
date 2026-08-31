import os

from photo2fcstd import thresholds as th

LEARNED = os.environ.get("P2F_LEARNED_MODES") == "1"


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


def learned_mode(specs):
    from photo2fcstd import mode_model, telemetry
    return mode_model.predict([telemetry.view_event(v) for v in specs])


def can_build(mode, specs):
    if mode == "revolve":
        return any(v["shape"].get("roundish") for v in specs)
    return True


def source_for(mode, specs):
    holed = max(specs, key=lambda v: v["shape"]["hole_frac"])
    if mode == "revolve":
        return max([v for v in specs if v["shape"].get("roundish")], key=lambda v: v["shape"]["ellipse"]["aspect"])
    if mode == "profile":
        least_rect = min(specs, key=lambda v: v["shape"]["rectangularity"])
        return holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else least_rect
    if mode == "plan":
        return holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else specs[0]
    return specs[0]


def select(specs, forced=None):
    if not specs:
        raise ValueError("no views to choose a mode from: the part has no photos")
    if forced is None and LEARNED:
        predicted = learned_mode(specs)
        if predicted and can_build(predicted, specs):
            return predicted, source_for(predicted, specs)
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


def edge_on_depth(src, views):
    aspects = [(min(v["shape"]["bbox"]) / max(v["shape"]["bbox"]), v) for v in views if max(v["shape"]["bbox"])]
    if not aspects:
        return None
    aspect, view = min(aspects, key=lambda t: t[0])
    return (th.DEPTH_FROM_ASPECT * aspect * src["length_px"],
            "depth estimated from the most edge-on photo (%s, aspect %.2f) - typically within 2x, measure it (px units)"
            % (os.path.basename(view["source"]), aspect))


def predicted_depth(src, others):
    from photo2fcstd import depth_model
    ratio = depth_model.ratio([src] + list(others))
    if ratio is None:
        return None
    return (ratio * src["length_px"],
            "depth predicted from the silhouettes at %.3f of length - typically within 2x, measure it (px units)" % ratio)


def outline_depth(src, others, mode, thickness_px):
    if thickness_px is not None:
        return thickness_px, "thickness from --thickness-px (px units)"
    if mode == "plan":
        learned = predicted_depth(src, others)
        if learned is not None:
            return learned
        return (th.PLATE_FRAC * src["length_px"],
                "plate thickness: NOT visible when every photo shows the same face, guessed as %.0f%% of length - set it from a caliper (px units)"
                % (th.PLATE_FRAC * 100))
    extent = max(src["shape"]["bbox"])
    other = max(others, key=lambda v: v["shape"]["rectangularity"]) if others else None
    if other is not None and other["shape"]["rectangularity"] > th.PLAN_RECT:
        width = max(st["width"] for st in other["stations"])
        return (width * extent / other["length_px"],
                "extrusion length = width of %s scaled by the profile extent (px units)" % os.path.basename(other["source"]))
    learned = predicted_depth(src, others)
    if learned is not None:
        return learned
    measured = edge_on_depth(src, [src] + list(others))
    if measured is not None:
        return measured
    return (min(src["shape"]["stroke_px"], th.PLATE_FRAC * src["length_px"]),
            "no plain elevation photo: depth guessed as min(section stroke, %.0f%% of length) - measure it (px units)"
            % (th.PLATE_FRAC * 100))
