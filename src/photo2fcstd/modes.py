import os

from photo2fcstd import thresholds as th

LEARNED = os.environ.get("P2F_LEARNED_MODES") == "1"
VIEW_PICK = os.environ.get("P2F_VIEW_PICK", "first")
ORACLE_DEPTH = os.environ.get("P2F_ORACLE_DEPTH") == "1"
_TRUE_RATIOS = {}


def true_ratio(source):
    if not _TRUE_RATIOS:
        import json
        from photo2fcstd.settings import PACKAGE_ROOT
        path = os.path.join(PACKAGE_ROOT, "data", "depth_rows.json")
        rows = json.load(open(path)) if os.path.exists(path) else []
        _TRUE_RATIOS.update({r["part"]: r["ratio"] for r in rows})
    return _TRUE_RATIOS.get(os.path.basename(source).split("_")[0])


def oracle_depth(src):
    ratio = true_ratio(src["source"])
    return None if ratio is None else (ratio * src["length_px"],
                                       "depth from the STEP file (oracle experiment, px units)")


def pick_view(specs):
    """The hand-written fallbacks. The learned selectors live in `outline_source`, which is the
    one place that decision is made; putting the ranker here as well meant two models raced for it
    and the loser's answer was silently thrown away."""
    if VIEW_PICK == "largest":
        return max(specs, key=lambda v: v["shape"]["bbox"][0] * v["shape"]["bbox"][1])
    if VIEW_PICK == "rectangular":
        return max(specs, key=lambda v: v["shape"]["rectangularity"])
    if VIEW_PICK == "symmetric":
        return max(specs, key=lambda v: (len(v["symmetric"]), v["shape"]["solidity"]))
    if VIEW_PICK == "solid":
        return max(specs, key=lambda v: v["shape"]["solidity"])
    return specs[0]


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
    from photo2fcstd import mode_model, mode_pixels, telemetry
    allowed = [m for m in ("profile", "plan", "revolve") if can_build(m, specs)]
    from_pixels = mode_pixels.predict(specs, allowed)
    if from_pixels is not None:
        return from_pixels
    return mode_model.predict([telemetry.view_event(v) for v in specs], allowed)


def can_build(mode, specs):
    if mode == "revolve":
        return any(v["shape"].get("roundish") for v in specs)
    return True


CIRCLE_ASPECT = float(os.environ.get("P2F_CIRCLE_ASPECT", 0.95))


def circular(specs):
    """A view that is a circle, not merely roundish: a disc is one circle, so say so early.

    Discs were being routed to profile or plan and drawn as twenty to fifty line segments
    while a forced revolve gave a single circle scoring 1.00.
    """
    circles = [v for v in specs if v["shape"].get("round") and v["shape"].get("ellipse")
               and v["shape"]["ellipse"]["aspect"] >= CIRCLE_ASPECT]
    return max(circles, key=lambda v: v["shape"]["ellipse"]["aspect"]) if circles else None


def roundest(specs):
    """The view whose fitted ellipse is closest to a circle, or None if no view is round.

    Revolve needs a view carrying an ellipse fit; asking for one that has none raised a
    KeyError, which is how forcing revolve on a cylinder failed outright.
    """
    round_ones = [v for v in specs if v["shape"].get("roundish") and v["shape"].get("ellipse")]
    return max(round_ones, key=lambda v: v["shape"]["ellipse"]["aspect"]) if round_ones else None


def source_for(mode, specs):
    holed = max(specs, key=lambda v: v["shape"]["hole_frac"])
    if mode == "revolve":
        return roundest(specs) or specs[0]
    if mode == "profile":
        least_rect = min(specs, key=lambda v: v["shape"]["rectangularity"])
        return holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else least_rect
    if mode == "plan":
        return holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else pick_view(specs)
    return pick_view(specs)


USE_VIEW_MODEL = os.environ.get("P2F_VIEW_MODEL", "1") != "0"


EDGE_ON_RATIO = float(os.environ.get("P2F_EDGE_ON_RATIO", 2.95755))


def stat(view, key):
    """A selection statistic, taken before regularisation when the view carries one.

    Selection used to read the statistics that tracing produces, so a change to symmetry or
    simplification moved which photograph got traced; three pixels of sliver once cost a part
    0.95 down to 0.45 that way.
    """
    picked = view.get("select") or {}
    if key in picked:
        return picked[key]
    return view["elongation"] if key == "elongation" else view["shape"][key]


def not_edge_on(chosen, specs, ratio=EDGE_ON_RATIO):
    """Veto a view that is far more elongated than the flattest available."""
    allowed = face_on(specs, ratio)
    if chosen in allowed or not allowed:
        return chosen
    return min(allowed, key=lambda v: v["elongation"])


def face_on(specs, ratio=EDGE_ON_RATIO):
    """Drop views that are far more elongated than the flattest one - those are edge-on.

    A flat part photographed on its edge gives a sliver whose outline is not the shape of the
    part, and no amount of tracing recovers it. Part 00901 was traced from a view at elongation
    22 while two views at 3 were available.
    """
    if len(specs) < 2:
        return specs
    flattest = min(stat(v, "elongation") for v in specs)
    kept = [v for v in specs if stat(v, "elongation") <= ratio * max(flattest, 1e-6)]
    return kept or specs


def outline_source(specs, fallback):
    """Which photo to draw the outline from - the single entry point for that decision.

    Two learned selectors exist for it. `view_rank` was reachable only through
    `P2F_VIEW_PICK=ranker`, and this function used to run `view_model` over the top of whatever
    `pick_view` returned, so the ranker computed a view that was then discarded. On 261
    discriminating parts they are indistinguishable - view_rank minus view_model is
    +0.008 [-0.018, +0.034] and they agree with each other on 61% of parts - so there is no winner
    to keep, only a shadowing to remove. `P2F_VIEW_PICK=ranker` now actually selects the ranker.

    The rules below pick the worst of three views 27% of the time. Scoring each view the way the
    carve axis is scored and taking the best is worth +0.038 [+0.019, +0.056] of sketch IoU on
    250 held-out parts, lifting skill over a drawn circle from 0.148 to 0.221, with the same
    44 of 45 valid solids through FreeCAD. Set P2F_VIEW_MODEL=0 to go back to the rules.

    Undoing the viewpoint tilt would be worth about the same, +0.090, but nothing in a silhouette
    can find that warp - six criteria all score below leaving the photo alone. This is the half of
    the capture problem that is a choice between real photographs rather than a quantity the mask
    does not contain.
    """
    if len(specs) < 2:
        return fallback
    if VIEW_PICK == "ranker":
        from photo2fcstd import view_rank
        ranked = view_rank.best(specs)
        return not_edge_on(ranked if ranked is not None else fallback, specs)
    if not USE_VIEW_MODEL:
        return fallback
    from photo2fcstd import view_model
    i = view_model.choose(specs)
    return not_edge_on(specs[i] if i is not None else fallback, specs)


def select(specs, forced=None):
    if not specs:
        raise ValueError("no views to choose a mode from: the part has no photos")
    if forced is None:
        circle_view = circular(specs)
        if circle_view is not None:
            return "revolve", circle_view
    if forced is None and (LEARNED or os.environ.get("P2F_MODE_PIXELS", "1") == "1"):
        predicted = learned_mode(specs)
        if predicted and can_build(predicted, specs):
            source = source_for(predicted, specs)
            return predicted, (source if predicted == "revolve" else outline_source(specs, source))
    holed = max(specs, key=lambda v: v["shape"]["hole_frac"])
    least_rect = min(specs, key=lambda v: v["shape"]["rectangularity"])
    if forced == "profile":
        return "profile", holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else least_rect
    if forced == "revolve":
        return "revolve", roundest(specs) or (holed if holed["shape"]["hole_frac"] > th.HOLE_FRAC_VISIBLE else specs[0])
    if forced == "plan":
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
        return ("plan" if flat else "profile"), outline_source(specs, holed)
    if flat:
        return "plan", outline_source(specs, pick_view(specs))
    return "profile", outline_source(specs, least_rect)


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
    got = depth_model.predict([src] + list(others))
    if got is None:
        return None
    ratio, lo, hi, coverage, per_part = got
    depth = ratio * src["length_px"]
    if lo is None or hi is None:
        return depth, "depth predicted from the silhouettes at %.3f of length - measure it (px units)" % ratio
    spread = hi / max(lo, 1e-9)
    verdict = "good enough to build from" if spread < 2.0 else "too wide to trust, put a caliper on it"
    band = ("%.0f%% of the time between %.1f and %.1f (%.1fx spread, %s)"
            % (100 * coverage, lo * src["length_px"], hi * src["length_px"], spread, verdict)
            if per_part else
            "%.0f%% of parts land within %.1fx of this - a fixed calibration, not this part's own "
            "uncertainty, so %s" % (100 * coverage, spread, verdict))
    return depth, "depth predicted from the silhouettes: %.1f px, %s" % (depth, band)


def outline_depth(src, others, mode, thickness_px):
    if thickness_px is not None:
        return thickness_px, "thickness from --thickness-px (px units)"
    if ORACLE_DEPTH:
        known = oracle_depth(src)
        if known is not None:
            return known
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
