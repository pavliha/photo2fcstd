"""Training data for the breakpoint model: contours in, span boundaries and types out.

The model decides only what the geometry cannot - where a loop's breakpoints are and whether each
span is straight or curved. Corner positions are contour-point indices, so localisation inherits
the contour's own precision, and all fitting stays geometric downstream. Labels come from geometry
already trusted (rule 3): each resampled contour point is owned by the nearest ground-truth
element; breakpoints are ownership transitions.

Sources: Fusion 360 Gallery designs plus the PrintCAD *tuning* split only - the test split never
enters. Split by design/part (rule 2); noisy-ownership samples dropped (rule 5).
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd.synth import (CANVAS, LABEL, apply_h, fit_canvas, homography,  # noqa: E402
                               loop_edges, rasterise, segment_distance)

N_POINTS = 192
MAX_SPANS = 48
MAX_OWN_DIST = 2.5
MIN_SPAN = 3


def resample(contour, n=N_POINTS):
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, np.concatenate([contour[:, 0], contour[:1, 0]]))
    y = np.interp(t, s, np.concatenate([contour[:, 1], contour[:1, 1]]))
    return np.column_stack([x, y])


def owners_of(pts, elements):
    dists = np.stack([np.min(np.stack([segment_distance(pts, p[i], p[i + 1])
                                       for i in range(len(p) - 1)]), axis=0)
                      for _, p in elements])
    return np.argmin(dists, axis=0), np.min(dists, axis=0)


def spans_of(owner, types):
    n = len(owner)
    breaks = [i for i in range(n) if owner[i] != owner[i - 1]]
    if not breaks:
        return [(n - 1, types[owner[0]])], 0
    start = breaks[0]
    rolled = [(b - start) % n for b in breaks[1:]] + [n]
    spans, prev = [], 0
    for end in sorted(rolled):
        o = owner[(start + prev) % n]
        spans.append([end - 1, types[o]])
        prev = end
    merged = []
    for e, t in spans:
        if merged and e - (merged[-1][0] if not merged else -1) < 1:
            continue
        if merged and (e - merged[-1][0]) < MIN_SPAN:
            merged[-1][0] = e
            continue
        merged.append([e, t])
    if len(merged) > 1 and merged[-1][0] - merged[-2][0] < MIN_SPAN:
        merged[-2][0] = merged[-1][0]
        merged.pop()
    return [(e, t) for e, t in merged], start


def one(args):
    record, seed = args
    from photo2fcstd.trace import outline
    rng = np.random.default_rng(seed)
    loops = [loop_edges(lp) for lp in record["loops"]]
    loops = [lp for lp in loops if lp]
    if not loops:
        return None
    H = homography(rng)
    loops = [[(t, apply_h(H, p)) for t, p in lp] for lp in loops]
    loops = fit_canvas(loops, CANVAS)
    mask = rasterise(loops, CANVAS)
    if mask.sum() < 500:
        return None
    try:
        poly, shape = outline(mask)
    except Exception:
        return None
    contour = np.asarray(shape["raw"], float)
    if len(contour) < N_POINTS // 2:
        return None
    pts = resample(contour)
    elements = loops[0]
    owner, dist = owners_of(pts, elements)
    if np.median(dist) > MAX_OWN_DIST:
        return None
    types = [LABEL.get(t, 0) for t, _ in elements]
    spans, start = spans_of(owner, types)
    if len(spans) > MAX_SPANS:
        return None
    r = int(rng.integers(N_POINTS))
    pts = np.roll(pts, r - start, axis=0)
    spans = sorted(((e + r) % N_POINTS, t) for e, t in spans)
    centred = pts - pts.mean(axis=0)
    scale = float(np.abs(centred).max()) or 1.0
    return (centred / scale).astype(np.float32), spans


def build(out="data/seq_data.npz", per_source=8, limit=None, seed=0):
    from photo2fcstd.sketch_score import trustworthy
    fusion = json.load(open(os.path.join(ROOT, "data", "fusion360_ideal_sketches.json")))
    printcad = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    tune = set(open(os.path.join(ROOT, "data", "tune_ids.txt")).read().split())
    sources = [("f/" + k, fusion[k]) for k in sorted(fusion)]
    sources += [("p/" + k, printcad[k]) for k in sorted(printcad)
                if k in tune and trustworthy(printcad[k])]
    rng = np.random.default_rng(seed)
    if limit:
        limit = int(limit)
        sources = [sources[i] for i in rng.choice(len(sources), limit, replace=False)]
    jobs = [(rec, int(rng.integers(1 << 31)), name)
            for name, rec in sources for _ in range(per_source)]
    with Pool(8) as pool:
        results = pool.map(one, [(r, s) for r, s, _ in jobs])
    X, Y, G = [], [], []
    for (r, s, name), res in zip(jobs, results):
        if res is None:
            continue
        X.append(res[0])
        Y.append(res[1])
        G.append(name)
    flat = [[v for e, t in y for v in (int(t), int(e))] + [-1] for y in Y]
    L = max(len(f) for f in flat)
    seqs = np.full((len(flat), L), -2, np.int32)
    for i, f in enumerate(flat):
        seqs[i, :len(f)] = f
    np.savez_compressed(os.path.join(ROOT, out), X=np.stack(X), seq=seqs,
                        group=np.array(G))
    n_spans = [len(y) for y in Y]
    curved = np.mean([t for y in Y for _, t in y])
    print("%d samples from %d sources (%.0f%% yield), spans/sample median %d max %d, %.0f%% curved"
          % (len(X), len(sources), 100 * len(X) / max(len(jobs), 1),
             int(np.median(n_spans)), max(n_spans), 100 * curved))


if __name__ == "__main__":
    build(*(sys.argv[1:] and [sys.argv[1]] or []),
          **dict(kv.split("=") for kv in sys.argv[2:]) if len(sys.argv) > 2 else {})
