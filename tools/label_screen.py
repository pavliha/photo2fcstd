"""Can the archive label itself? Align each part's ideal sketch onto its real traced contour.

The seqnet post-mortem says the model path needs real-photo contours with known sketches. The
archive holds both halves; the question is whether similarity alignment connects them well enough
to be a label. Gate: chamfer cost after the best rotation x per-axis-scale x flip. Quality: the
median distance from contour points to their owning ideal element - the same statistic synth.py
calls label quality, where "well under a pixel" was the bar on synthetic input.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tracer_ceiling as TC  # noqa: E402
from arc_survival import transform_ideal  # noqa: E402
from photo2fcstd.synth import LABEL, segment_distance  # noqa: E402

RUN = "runs/seq_off"


def resample(contour, n=256):
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, np.concatenate([contour[:, 0], contour[:1, 0]])),
                            np.interp(t, s, np.concatenate([contour[:, 1], contour[:1, 1]]))])


def fit_once(pts, ideal_sub):
    lo_d, hi_d = pts.min(0), pts.max(0)
    lo_i, hi_i = ideal_sub.min(0), ideal_sub.max(0)
    sc = (hi_i - lo_i) / np.maximum(hi_d - lo_d, 1e-9)
    moved = (pts - lo_d) * sc + lo_i
    cost = float(np.mean(np.min(np.linalg.norm(moved[:, None] - ideal_sub[None], axis=2), axis=1)))
    return cost, sc, lo_d, lo_i


def align_rot(contour, ideal_px, coarse=24):
    ideal_sub = ideal_px[::max(len(ideal_px) // 400, 1)]
    c = contour - contour.mean(0)
    best = None
    for flip in (1.0, -1.0):
        base = c * [1.0, flip]
        for k in range(coarse):
            th = 2 * np.pi * k / coarse
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            cost, sc, lo_d, lo_i = fit_once(base @ R.T, ideal_sub)
            if best is None or cost < best[0]:
                best = (cost, flip, th)
    _, flip, th0 = best
    for th in np.linspace(th0 - np.pi / coarse, th0 + np.pi / coarse, 9):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        cost, sc, lo_d, lo_i = fit_once((c * [1.0, flip]) @ R.T, ideal_sub)
        if cost < best[0]:
            best = (cost, flip, th)
    cost, flip, th = best
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (c * [1.0, flip]) @ R.T
    _, sc, lo_d, lo_i = fit_once(pts, ideal_sub)
    return cost, (pts - lo_d) * sc + lo_i


def one(part):
    from photo2fcstd.trace import outline, segment_photo, upright_mask
    try:
        spec = json.load(open(os.path.join(ROOT, RUN, "out", part + ".spec.json")))
        ol = spec.get("outline")
        if not ol:
            return part, {"skip": "no outline mode"}
        mask, _ = upright_mask(segment_photo(ol["source"]))
        _, sh = outline(mask)
        contour = resample(np.asarray(sh["raw"], float))
        rec = TC.IDEAL[part]
        to_px, _ = transform_ideal(rec)
        elements = [(LABEL.get(e["type"], 0), to_px(TC.polyline(e))) for e in rec["loops"][0]]
        ideal_px = np.concatenate([p for _, p in elements])
        cost, moved = align_rot(contour, ideal_px)
        dists = np.stack([np.min(np.stack([segment_distance(moved, p[i], p[i + 1])
                                           for i in range(len(p) - 1)]), axis=0)
                          for _, p in elements])
        owner = np.argmin(dists, axis=0)
        own = np.min(dists, axis=0)
        types = np.array([t for t, _ in elements])
        return part, {"cost": cost, "median_own": float(np.median(own)),
                      "p90_own": float(np.percentile(own, 90)),
                      "curved_frac": float(np.mean(types[owner])),
                      "moved": moved.tolist(), "owner_type": types[owner].tolist()}
    except Exception as e:
        return part, {"skip": str(e)[:60]}


def main(limit=60):
    from photo2fcstd import sketch_score as SS
    tune = set(open(os.path.join(ROOT, "data", "tune_ids.txt")).read().split())
    parts = [p for p in sorted(TC.IDEAL) if p in tune and SS.trustworthy(TC.IDEAL[p])
             and os.path.exists(os.path.join(ROOT, RUN, "out", p + ".spec.json"))][:limit]
    with Pool(6) as pool:
        rows = dict(pool.map(one, parts))
    ok = {p: v for p, v in rows.items() if "cost" in v}
    print("parts: %d fed, %d with an outline trace, %d skipped (%s)" % (
        len(parts), len(ok), len(parts) - len(ok),
        ", ".join(sorted({v["skip"] for v in rows.values() if "skip" in v})[:3])))
    costs = np.array([v["cost"] for v in ok.values()])
    print("\n  alignment cost (ideal frame is 900 px, part spans 720):")
    for thr in (8, 12, 18, 25):
        sel = {p: v for p, v in ok.items() if v["cost"] <= thr}
        if not sel:
            print("  cost <= %2d: 0 parts" % thr)
            continue
        med = np.median([v["median_own"] for v in sel.values()])
        p90 = np.median([v["p90_own"] for v in sel.values()])
        print("  cost <= %2d: %2d parts (%.0f%%)  label error median %.1f px, p90 %.1f px  (%.2f%% of length)"
              % (thr, len(sel), 100 * len(sel) / len(parts), med, p90, 100 * med / 720))
    json.dump({p: {k: v for k, v in r.items() if k != "moved"} for p, r in ok.items()},
              open(os.path.join(ROOT, "data", "label_screen.json"), "w"))
    passing = sorted(ok, key=lambda p: ok[p]["cost"])[:6]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.2))
    for a, p in zip(ax, passing):
        m = np.array(ok[p]["moved"])
        t = np.array(ok[p]["owner_type"])
        a.scatter(m[:, 0], m[:, 1], c=["C2" if x else "C0" for x in t], s=4)
        rec_pts = np.concatenate([TC.polyline(e) for e in TC.IDEAL[p]["loops"][0]])
        to_px, _ = transform_ideal(TC.IDEAL[p])
        q = to_px(rec_pts)
        a.plot(q[:, 0], q[:, 1], "k-", lw=0.5, alpha=0.5)
        a.set_title("%s cost %.1f" % (p, ok[p]["cost"]), fontsize=8)
        a.set_aspect("equal"); a.invert_yaxis(); a.axis("off")
    fig.suptitle("real contour (dots, coloured by inherited label) aligned onto the ideal sketch (grey)", fontsize=10)
    fig.savefig("/private/tmp/claude-501/-Users-pavliha-Code-photo2fcstd/645d85db-e384-4236-9a84-ba2c794f26e8/scratchpad/label_screen.png",
                dpi=80, bbox_inches="tight")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
