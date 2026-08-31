import itertools
import json
import os

import numpy as np

from photo2fcstd import stats

KEYS = ("rect", "solidity", "elongation", "hole_frac", "ellipse_rms")
SOURCE = os.path.join(os.path.dirname(__file__), "..", "data", "view_features.json")


def scales(recs):
    return {k: (np.array([s[k] for r in recs for s in r["shots"]], float).std() or 1.0) for k in KEYS}


def distance(a, b, sd):
    return float(np.sqrt(sum(((a[k] - b[k]) / sd[k]) ** 2 for k in KEYS)))


def spread(rec, sd):
    pairs = itertools.combinations(rec["shots"], 2)
    return max((distance(a, b, sd) for a, b in pairs), default=0.0)


def inflation(recs, threshold):
    sd = scales(recs)
    twins = [r for r in recs if len(r["shots"]) == 3 and spread(r, sd) < threshold]
    gains = [max(s["iou"] for s in r["shots"]) - r["shots"][0]["iou"] for r in twins]
    return len(twins), stats.mean_ci(gains)


def report(recs):
    gap = stats.mean_ci([max(s["iou"] for s in r["shots"]) - r["shots"][0]["iou"] for r in recs])
    rows = [(t,) + inflation(recs, t) for t in (0.4, 0.6, 0.8)]
    return gap, rows


def main():
    recs = json.load(open(os.path.normpath(SOURCE)))
    gap, rows = report(recs)
    print("oracle over 3 views, all %d parts: %+.3f [%+.3f, %+.3f]" % ((len(recs),) + gap))
    for threshold, n, (m, lo, hi) in rows:
        print("  of which luck (interchangeable views, spread<%.1f, n=%3d): %+.3f [%+.3f, %+.3f]"
              % (threshold, n, m, lo, hi))
    _, n, (m, _, _) = rows[1]
    print("real viewpoint effect: about %+.3f; the ranker takes %+.3f of it" % (gap[0] - m, 0.031))


if __name__ == "__main__":
    main()
