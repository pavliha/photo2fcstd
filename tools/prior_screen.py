"""Does a design prior fitted on CAD alone pick the better of two candidate drawings?

The noise-floor diagnosis says ambiguous decisions cannot be settled from the contour. The Bayes
fix says a prior over sketches settles them instead. This is its cheapest test: every measured A/B
left two candidate drawings per changed part - shipped and variant, both fitting the same contour -
and the ideal says which is truly better. A count-based prior over loop compositions, fitted on
tuning-split ideals with the evaluated part left out, scores both. If it cannot beat the best fixed
policy here, no generative model will.
"""
import collections, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
CAP = 6


def key_of_counts(counter):
    return tuple(sorted((t, min(c, CAP)) for t, c in counter.items()))


def ideal_loop_keys(part):
    return [key_of_counts(collections.Counter(e["type"] for e in loop))
            for loop in (IDEAL[part].get("loops") or [])]


def spec_loop_keys(spec):
    out = []
    rv = spec.get("revolve")
    if rv and not spec.get("outline"):
        out.append(key_of_counts(collections.Counter({"circle": 1})))
        for h in rv.get("holes", []):
            out.append(key_of_counts(collections.Counter({h.get("type", "circle"): 1})))
        return out
    for loop in (spec.get("outline") or {}).get("loops", []):
        if loop["type"] in ("circle", "ellipse"):
            out.append(key_of_counts(collections.Counter({loop["type"]: 1})))
        else:
            out.append(key_of_counts(collections.Counter(e["type"] for e in loop["elements"])))
    return out


def main():
    from photo2fcstd.sketch_score import trustworthy
    tune = set(open(os.path.join(ROOT, "data", "tune_ids.txt")).read().split())
    prior_parts = [p for p in sorted(IDEAL) if p in tune and trustworthy(IDEAL[p])]
    counts = collections.Counter()
    per_part = {}
    for p in prior_parts:
        ks = ideal_loop_keys(p)
        per_part[p] = collections.Counter(ks)
        counts.update(ks)
    N = sum(counts.values())
    V = len(counts) + 1
    alpha = 0.5

    def logp(keys, exclude=None):
        c = counts.copy()
        n = N
        if exclude in per_part:
            c.subtract(per_part[exclude])
            n -= sum(per_part[exclude].values())
        return sum(math.log((c[k] + alpha) / (n + alpha * V)) for k in keys)

    pairs = []
    for name, json_path, runs, arms in (
            ("tangent", "data/ab_tangent.json", ("tang_off", "tang_on"), ("off", "on")),
            ("bspline", "data/ab_bspline.json", ("bspl_off", "bspl_on"), ("off", "on")),
            ("holegate", "data/ab_hole_gate.json", ("holegate_0", "holegate_5"), ("0", "5"))):
        by = json.load(open(os.path.join(ROOT, json_path)))
        a, b = arms
        for part in by[a]:
            va, vb = by[a][part], by[b].get(part, {})
            if "f1" not in va or "f1" not in vb or va["f1"] == vb["f1"]:
                continue
            sa = os.path.join(ROOT, "runs", runs[0], "out", part + ".spec.json")
            sb = os.path.join(ROOT, "runs", runs[1], "out", part + ".spec.json")
            if not (os.path.exists(sa) and os.path.exists(sb)):
                continue
            ka = spec_loop_keys(json.load(open(sa)))
            kb = spec_loop_keys(json.load(open(sb)))
            if ka == kb:
                continue
            pairs.append({"ab": name, "part": part, "f1": (va["f1"], vb["f1"]),
                          "lp": (logp(ka, part), logp(kb, part))})

    print("decision pairs with differing truth and differing composition: %d" % len(pairs))
    for name in ("tangent", "bspline", "holegate", "ALL"):
        sel = [q for q in pairs if q["ab"] == name or name == "ALL"]
        if not sel:
            continue
        right = sum((q["lp"][1] > q["lp"][0]) == (q["f1"][1] > q["f1"][0]) for q in sel)
        base_a = sum(q["f1"][0] > q["f1"][1] for q in sel)
        base_b = len(sel) - base_a
        print("  %-8s n=%2d  prior picks better drawing %2d (%3.0f%%)   fixed policies: %.0f%% / %.0f%%"
              % (name, len(sel), right, 100 * right / len(sel),
                 100 * base_a / len(sel), 100 * base_b / len(sel)))
    print("\nper pair (truthfully-better arm marked >):")
    for q in sorted(pairs, key=lambda q: q["ab"]):
        better_b = q["f1"][1] > q["f1"][0]
        pick_b = q["lp"][1] > q["lp"][0]
        print("  %-8s %s  f1 %.2f%s/ %.2f%s  logp %7.2f / %7.2f  %s"
              % (q["ab"], q["part"], q["f1"][0], " " if better_b else ">",
                 q["f1"][1], ">" if better_b else " ",
                 q["lp"][0], q["lp"][1], "OK" if pick_b == better_b else "wrong"))


if __name__ == "__main__":
    main()
