"""How clean is the geometry we emit, regardless of whether it matches the truth?"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from photo2fcstd import stats

POPULAR = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], float)


def angle_of(e):
    d = np.subtract(e["p1"], e["p0"])
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0), float(np.hypot(*d))


def loops_of(run, part):
    path = os.path.join(run, "out", part + ".spec.json")
    if not os.path.exists(path):
        return []
    doc = json.load(open(path))
    return ((doc.get("outline") or {}).get("loops") or [])


def measure(run, parts):
    from photo2fcstd.trace import dominant_frame
    clean, right, tiny, counts = [], [], [], []
    for part in parts:
        angles, lengths, elements, frames = [], [], 0, []
        for loop in loops_of(run, part):
            if loop.get("type") == "circle":
                continue
            els = [e for e in loop.get("elements", []) if e["type"] == "line"]
            elements += len(loop.get("elements", []))
            if len(els) >= 3:
                pts = np.array([e["p0"] for e in els], float)
                weight = sum(np.hypot(*np.subtract(e["p1"], e["p0"])) for e in els)
                frames.append((weight, dominant_frame(pts)))
            for e in els:
                a, L = angle_of(e)
                angles.append(a)
                lengths.append(L)
        if not angles:
            continue
        angles = np.array(angles)
        frame = max(frames)[1] if frames else 0.0
        targets = (POPULAR + frame) % 180.0
        gap = np.abs(angles[:, None] - targets[None, :])
        gap = np.minimum(gap, 180.0 - gap).min(axis=1)
        clean.append(float(np.mean(gap <= 2.0)))
        span = max(np.sum(lengths), 1.0)
        tiny.append(float(np.sum([L for L in lengths if L < 0.02 * span]) / span))
        pairs = np.abs(np.diff(np.concatenate([angles, angles[:1]])))
        pairs = np.minimum(pairs, 180 - pairs)
        right.append(float(np.mean(np.abs(pairs - 90) <= 3.0)))
        counts.append(elements)
    return clean, right, tiny, counts


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/final"
    parts = sorted({f.split(".")[0] for f in os.listdir(os.path.join(run, "out")) if f.endswith(".spec.json")})
    clean, right, tiny, counts = measure(run, parts)
    show = lambda label, xs: print("  %-34s %.3f [%.3f, %.3f]" % ((label,) + tuple(stats.mean_ci(xs))))
    print("%s: %d parts with line work" % (run, len(clean)))
    show("segments on a popular angle", clean)
    show("adjacent pairs at a right angle", right)
    show("length in segments under 2% of the loop", tiny)
    print("  %-34s %.1f" % ("median elements per part", float(np.median(counts))))


if __name__ == "__main__":
    main()
