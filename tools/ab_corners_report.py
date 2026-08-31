import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import stats  # noqa: E402

A = json.load(open(os.path.join(ROOT, "data", "ab_corners_approxPolyDP.json")))
B = json.load(open(os.path.join(ROOT, "data", "ab_corners_learned.json")))
Cc = json.load(open(os.path.join(ROOT, "data", "ab_corners_filtered.json")))
both = [k for k in A if A.get(k) and B.get(k) and Cc.get(k)]
a = {k: A[k] for k in both}
b = {k: B[k] for k in both}
c = {k: Cc[k] for k in both}
keen = [k for k in both if 1.0 - a[k]["trivial"] >= 0.15]

print("%d parts scored by both arms, %d of them discriminating\n" % (len(both), len(keen)))
print("  %-16s %8s %8s %9s %8s %8s" % ("arm", "IoU", "skill", "exact", "loops", "curve"))
for name, d in (("approxPolyDP", a), ("learned corners", b), ("filtered", c)):
    v = np.array([d[k]["iou"] for k in keen])
    t = np.array([d[k]["trivial"] for k in keen])
    print("  %-16s %8.3f %8.3f %8.0f%% %8.1f %8.2f"
          % (name, v.mean(), (v.mean() - t.mean()) / (1 - t.mean()),
             100 * np.mean([d[k]["exact"] for k in keen]),
             np.mean([d[k]["loops"] for k in keen]),
             np.mean([d[k]["curve"] for k in keen])))
print("  %-16s %8.3f" % ("ideal loops", np.mean([a[k]["ideal_loops"] for k in keen])))

for name, d in (("learned corners", b), ("filtered", c)):
    dv = np.array([d[k]["iou"] - a[k]["iou"] for k in keen])
    m, lo, hi = stats.mean_ci(dv)
    print("\n  %s minus approxPolyDP: %+.4f [%+.4f, %+.4f]" % (name, m, lo, hi))
    print("  better on %d parts, worse on %d, unchanged on %d"
          % (int((dv > 1e-6).sum()), int((dv < -1e-6).sum()), int((np.abs(dv) <= 1e-6).sum())))
