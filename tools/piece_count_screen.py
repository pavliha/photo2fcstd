"""Does the number of straight pieces separate a noisy circle from a rectangle,
where the residual does not?"""
import json, os, sys, collections
sys.path.insert(0, "src")
import numpy as np
from photo2fcstd import bench
from photo2fcstd.trace import (fit_ellipse, outline, segment_photo, upright_mask, elements,
                               boundary_period, feature_amplitude, CIRCLE_VETO_AMPLITUDE)

RUN = "runs/tier_now"
IDEAL = json.load(open("data/printcad_ideal_sketches_all.json"))

def ideal_loop_kinds(part):
    r = IDEAL.get(part) or {}
    return [collections.Counter(e.get("type") for e in l) for l in (r.get("loops") or [])]

rows = []
for part, row in sorted(bench.sketch_scores(RUN).items()):
    if not row.get("trustworthy"): continue
    p = os.path.join(RUN, "out", part + ".spec.json")
    if not os.path.exists(p): continue
    ol = json.load(open(p)).get("outline")
    if not ol: continue
    ideal = ideal_loop_kinds(part)
    if len(ideal) < 2: continue
    try:
        mask, _ = upright_mask(segment_photo(ol["source"]))
        _, sh = outline(mask)
    except Exception: continue
    if len(sh["raw_holes"]) != len(ideal) - 1:
        continue
    order = np.argsort([-len(h) for h in sh["raw_holes"]])
    ideal_holes = sorted(ideal[1:], key=lambda c: -sum(c.values()))
    for rank, i in enumerate(order):
        raw = np.asarray(sh["raw_holes"][i], float)
        f = fit_ellipse(raw)
        if f is None or len(raw) < 12: continue
        try:
            k = len(elements(raw, float(np.hypot(*np.ptp(np.asarray(sh["raw"], float), axis=0)))))
        except Exception: continue
        want = ideal_holes[rank] if rank < len(ideal_holes) else collections.Counter()
        rows.append(dict(part=part, rel=f["rms"] / f["a"], a=f["a"], aspect=f["aspect"], k=k,
                         is_circle=set(want) == {"circle"}, want=dict(want)))

print("hole loops with a matched ideal loop: %d over %d parts\n" % (len(rows), len({r["part"] for r in rows})))
circ = [r for r in rows if r["is_circle"]]
poly = [r for r in rows if not r["is_circle"]]
print("  %-22s %5s %8s %8s %8s" % ("group", "n", "rms/a", "k median", "k mean"))
for label, sel in (("ideal loop is a circle", circ), ("ideal loop is not", poly)):
    if not sel: continue
    print("  %-22s %5d %8.3f %8.0f %8.1f" % (label, len(sel), np.median([r["rel"] for r in sel]),
                                             np.median([r["k"] for r in sel]), np.mean([r["k"] for r in sel])))
print()
missed = [r for r in circ if r["rel"] >= 0.04 and r["aspect"] > 0.42]
print("  circles the residual gate misses (rms/a >= 0.04): %d" % len(missed))
if missed:
    print("    their k: median %.0f, %s" % (np.median([r["k"] for r in missed]), sorted(r["k"] for r in missed)))
near = [r for r in poly if r["rel"] < 0.09]
print("  non-circles with rms/a < 0.09 (what an absolute rule would take): %d" % len(near))
if near:
    print("    their k: median %.0f, %s" % (np.median([r["k"] for r in near]), sorted(r["k"] for r in near)))
for thr in (5, 6, 7, 8):
    tp = sum(1 for r in missed if r["k"] >= thr)
    fp = sum(1 for r in near if r["k"] >= thr)
    print("  k >= %d  would recover %d of %d missed circles, and wrongly take %d of %d non-circles"
          % (thr, tp, len(missed), fp, len(near)))
