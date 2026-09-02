"""Choose which photographs to model from when more are offered than the pipeline uses.

The command line used to take the first three and silently drop the rest, so sending ten
photographs of a part meant modelling from whichever three came first in the shell glob.
Three views are wanted: one to draw the outline from, and two whose elevations differ enough
to measure a height. Picking them by hand is the difference between a measured dimension and
a guessed one.
"""
import os


def spread(views, count=3):
    """Keep the flattest view, then the ones least like what is already kept."""
    if len(views) <= count:
        return list(views)
    kept = [min(views, key=lambda v: signature(v)[0])]
    while len(kept) < count:
        rest = [v for v in views if v not in kept]
        if not rest:
            break
        kept.append(max(rest, key=lambda v: min(distance(v, k) for k in kept)))
    return kept


def signature(view):
    picked = view.get("select") or {}
    shape = view.get("shape") or {}

    def get(key, fallback):
        if key in picked:
            return float(picked[key])
        if key == "elongation":
            return float(view.get("elongation", fallback))
        return float(shape.get(key, fallback))

    return (get("elongation", 1.0), get("rectangularity", 0.9),
            get("solidity", 0.95), get("hole_frac", 0.0))


def distance(a, b):
    x, y = signature(a), signature(b)
    scales = (4.0, 0.4, 0.2, 0.3)
    return sum(abs(p - q) / s for p, q, s in zip(x, y, scales))


def describe(chosen, offered):
    if len(offered) <= len(chosen):
        return ""
    names = ", ".join(os.path.basename(v["source"]) for v in chosen)
    return "modelling from %d of %d photographs: %s" % (len(chosen), len(offered), names)
