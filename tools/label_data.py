"""Real-photo training samples: the archive labelling itself through alignment.

Every sample is a real traced contour from a *tuning-split* photograph, its labels inherited from
the part's ideal sketch through the gated alignment that tools/label_screen.py measured at 2.6 px
median error and 55% yield. Each accepted contour is emitted several times under a random roll and
rotation so orientation is not learnable. The test split never enters.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tracer_ceiling as TC  # noqa: E402
from arc_survival import transform_ideal  # noqa: E402
from label_screen import align_rot, resample  # noqa: E402
from seq_data import MAX_SPANS, N_POINTS, spans_of  # noqa: E402
from photo2fcstd.synth import LABEL, segment_distance  # noqa: E402

COST_GATE = 12.0
OWN_GATE = 4.5
REPEATS = 6


def one(args):
    part, photo = args
    from photo2fcstd.trace import outline, segment_photo, upright_mask
    try:
        mask, _ = upright_mask(segment_photo(photo))
        _, sh = outline(mask)
        contour = resample(np.asarray(sh["raw"], float), N_POINTS)
        rec = TC.IDEAL[part]
        to_px, _ = transform_ideal(rec)
        elements = [(LABEL.get(e["type"], 0), to_px(TC.polyline(e))) for e in rec["loops"][0]]
        ideal_px = np.concatenate([p for _, p in elements])
        cost, moved = align_rot(contour, ideal_px)
        if cost > COST_GATE:
            return []
        dists = np.stack([np.min(np.stack([segment_distance(moved, p[i], p[i + 1])
                                           for i in range(len(p) - 1)]), axis=0)
                          for _, p in elements])
        owner = np.argmin(dists, axis=0)
        if np.median(np.min(dists, axis=0)) > OWN_GATE:
            return []
        types = [t for t, _ in elements]
        spans, start = spans_of(owner, types)
        if len(spans) > MAX_SPANS:
            return []
        rng = np.random.default_rng(abs(hash(photo)) % (1 << 31))
        out = []
        for _ in range(REPEATS):
            r = int(rng.integers(N_POINTS))
            pts = np.roll(contour, r - start, axis=0)
            sp = sorted(((e + r) % N_POINTS, t) for e, t in spans)
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            q = (pts - pts.mean(axis=0)) @ R.T
            scale = float(np.abs(q).max()) or 1.0
            out.append(((q / scale).astype(np.float32), sp))
        return [(part, x, sp) for x, sp in out]
    except Exception:
        return []


def build(out="data/seq_real.npz", limit=None):
    from photo2fcstd import bench, sketch_score as SS
    tune = set(open(os.path.join(ROOT, "data", "tune_ids.txt")).read().split())
    parts = [p for p in sorted(TC.IDEAL) if p in tune and SS.trustworthy(TC.IDEAL[p])]
    parts = bench.with_photos(parts)[0]
    if limit:
        parts = parts[:limit]
    jobs = [(p, ph) for p in parts for ph in bench.photos_of(p)[:3]]
    with Pool(6) as pool:
        rows = [r for rs in pool.map(one, jobs) for r in rs]
    X = np.stack([x for _, x, _ in rows])
    G = np.array(["p/" + p for p, _, _ in rows])
    flat = [[v for e, t in sp for v in (int(t), int(e))] + [-1] for _, _, sp in rows]
    L = max(len(f) for f in flat)
    seqs = np.full((len(flat), L), -2, np.int32)
    for i, f in enumerate(flat):
        seqs[i, :len(f)] = f
    np.savez_compressed(os.path.join(ROOT, out), X=X, seq=seqs, group=G)
    print("%d real samples from %d photographs of %d parts (%.0f%% of photos passed the gates)"
          % (len(rows), len(rows) // REPEATS, len({p for p, _, _ in rows}),
             100 * (len(rows) // REPEATS) / max(len(jobs), 1)))


if __name__ == "__main__":
    build(*(sys.argv[1:2] or []), **({"limit": int(sys.argv[2])} if len(sys.argv) > 2 else {}))
