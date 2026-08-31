import os

from photo2fcstd import modes
from photo2fcstd import thresholds as th
from photo2fcstd.trace import primitives


def scale_of(views, mm_per_px, length_mm):
    if mm_per_px is not None:
        return mm_per_px, "from --mm-per-px (rectified photo or known scale)"
    if length_mm is not None:
        length = views["front"]["length_px"]
        return length_mm / length, "from --length-mm %s over %.1f px" % (length_mm, length)
    return 1.0, "UNSCALED: set this from one caliper reading (mm / px)"


def rescale_side(views):
    front, side = views["front"]["length_px"], views["side"]["length_px"]
    if not side:
        raise ValueError("side view has zero length: check the segmentation of %s" % views["side"]["source"])
    f = front / side
    views["side"]["z"] = [z * f for z in views["side"]["z"]]
    views["side"]["stations"] = [{"width": s["width"] * f} for s in views["side"]["stations"]]
    return f


def not_degenerate(e):
    return (e["p0"][0] != e["p1"][0] or e["p0"][1] != e["p1"][1]) and (e["type"] != "arc" or e["r"] > 0)


def rounded_loops(loops, rnd):
    out = []
    for loop in loops:
        if loop["type"] == "circle":
            out.append(dict(loop, cx=rnd(loop["cx"]), cy=rnd(loop["cy"]), r=rnd(loop["r"])))
            continue
        elements = [dict(e, p0=[rnd(e["p0"][0]), rnd(e["p0"][1])], p1=[rnd(e["p1"][0]), rnd(e["p1"][1])],
                         **({"cx": rnd(e["cx"]), "cy": rnd(e["cy"]), "r": rnd(e["r"])} if e["type"] == "arc" else {}))
                    for e in loop["elements"]]
        kept = [e for e in elements if not_degenerate(e)]
        elements = kept if len(kept) >= 2 else elements
        out.append(dict(loop, elements=elements, kinds=[k for k, e in zip(loop["kinds"], elements)] if len(elements) == len(loop["kinds"]) else loop["kinds"][:len(elements)],
                        joins=loop["joins"][:len(elements)]))
    return out


def merged_stations(view, rnd, floor):
    ws = [max(rnd(st["width"]), floor) for st in view["stations"]]
    keep = [i for i in range(len(ws)) if i == 0 or ws[i] != ws[i - 1]]
    return [{"width": ws[i]} for i in keep], [view["z"][i] for i in keep] + [view["z"][-1]]


def assemble(specs, name, mode=None, mm_per_px=None, length_mm=None, thickness_px=None, rim_px=None, stl=None, log=print):
    mode_sel, src = modes.select(specs, mode)
    others = [v for v in specs if v is not src]
    log("mode: %s   (rectangularity %s, elongation %s)"
        % (mode_sel, [round(v["shape"]["rectangularity"], 2) for v in specs], [v["elongation"] for v in specs]))
    outline_spec = revolve_spec = None
    if mode_sel == "revolve":
        ellipse = src["shape"]["ellipse"]
        loops = primitives(modes.centred_loops(src, ellipse["cx"], ellipse["cy"]), src["length_px"])
        R = loops[0]["r"] if loops[0]["type"] == "circle" else ellipse["a"]
        elevation = max([v for v in others if modes.is_elevation(v)], key=lambda v: v["shape"]["rectangularity"], default=None)
        if elevation is not None:
            revolve_spec = modes.revolve_from_elevation(src, elevation, loops, R)
            log("revolve: R=%.0f px, profile from %s, %d holes" % (R, os.path.basename(elevation["source"]), len(revolve_spec["holes"])))
        else:
            rings = sorted(src["shape"].get("rings", []), reverse=True)
            revolve_spec = modes.revolve_from_prior(src, loops, R, rings, thickness_px, rim_px)
            log("revolve: R=%.0f px, concentric edges at %s of R, %d holes; height is a prior"
                % (R, [round(r, 2) for r in rings], len(revolve_spec["holes"])))
        views = {"front": src}
    elif mode_sel in ("profile", "plan"):
        raw = src["shape"]["raw"]
        cx = sum(p[0] for p in raw) / len(raw)
        cy = sum(p[1] for p in raw) / len(raw)
        loops = primitives(modes.centred_loops(src, cx, cy), src["length_px"])
        depth, note = modes.outline_depth(src, others, mode_sel, thickness_px)
        outline_spec = {"source": src["source"], "loops": loops, "depth_px": depth, "depth_note": note}
        log("sketch: %s%s" % (", ".join("circle r=%.0f" % l["r"] if l["type"] == "circle"
                                        else "%d elements (%s)" % (len(l["elements"]), "".join(l["kinds"])) for l in loops),
                              "; symmetric about " + "".join(src["symmetric"]) if src["symmetric"] else ""))
        views = {"front": src}
    else:
        views, dropped, end_on = modes.station_views(specs)
        if dropped:
            log("dropped %s as the axial view" % os.path.basename(dropped))
        if end_on:
            log("WARNING: %s looks end-on; a second elevation would be better" % os.path.basename(end_on))
    mpp, scale_note = scale_of(views, mm_per_px, length_mm)
    if len(views) == 2:
        log("side view rescaled by %.3f to match front length" % rescale_side(views))
    q = 1.0 if mm_per_px is None and length_mm is None else th.ROUND_MM / mpp
    rnd = lambda v: round(round(v / q) * q, 4)
    if revolve_spec:
        revolve_spec["profile"] = [[rnd(x), rnd(y)] for x, y in revolve_spec["profile"]]
        revolve_spec["holes"] = rounded_loops(revolve_spec["holes"], rnd)
    if outline_spec:
        outline_spec["loops"] = rounded_loops(outline_spec["loops"], rnd)
        outline_spec["depth_px"] = rnd(outline_spec["depth_px"])
    for v in views.values():
        v["stations"], v["z"] = merged_stations(v, rnd, q)
    for n, v in (views.items() if not (outline_spec or revolve_spec) else []):
        log("%s: rotated %+.0f deg, %d stations over %.0f px  widths %s"
            % (n, v["angle_deg"], len(v["stations"]), v["length_px"], [round(s["width"], 1) for s in v["stations"]]))
    return {"name": name, "mm_per_px": mpp, "scale_note": scale_note,
            "views": {k: {kk: vv for kk, vv in v.items() if kk not in ("poly", "shape")} for k, v in views.items()},
            "outline": outline_spec, "revolve": revolve_spec, "stl": stl}
