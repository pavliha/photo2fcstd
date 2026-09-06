import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    base = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "runs", "gate_bench_base.json"))) if "region_iou" in r}
    fp = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "runs", "gate_bench_facepose.json"))) if "region_iou" in r}
    parts = sorted(set(base) & set(fp))
    rect = [p for p in parts if fp[p].get("rectified")]
    d = np.array([fp[p]["region_iou"] - base[p]["region_iou"] for p in rect])
    tilt = np.array([fp[p]["tilt"] for p in rect])
    print("PAIRED parts=%d rectified=%d | mean region IoU base %.3f -> facepose %.3f | on rectified parts: base %.3f -> %.3f (delta %+.3f, wins %d, losses %d)"
          % (len(parts), len(rect), np.mean([base[p]["region_iou"] for p in parts]), np.mean([fp[p]["region_iou"] for p in parts]),
             np.mean([base[p]["region_iou"] for p in rect]), np.mean([fp[p]["region_iou"] for p in rect]), d.mean(), int((d > 0.01).sum()), int((d < -0.01).sum())))
    for lo, hi in ((20, 30), (30, 40), (40, 90)):
        sel = (tilt >= lo) & (tilt < hi)
        if sel.sum():
            print("  tilt %2d-%2d: n=%2d delta %+.3f  wins %d losses %d  base %.3f" % (lo, hi, sel.sum(), d[sel].mean(), int((d[sel] > 0.01).sum()), int((d[sel] < -0.01).sum()), np.mean([base[p]["region_iou"] for p, s in zip(rect, sel) if s])))
    low = [p for p in rect if base[p]["region_iou"] < 0.85]
    if low:
        dl = np.array([fp[p]["region_iou"] - base[p]["region_iou"] for p in low])
        print("  where the photo path was weak (base < 0.85): n=%d delta %+.3f wins %d losses %d" % (len(low), dl.mean(), int((dl > 0.01).sum()), int((dl < -0.01).sum())))
    worst = sorted(rect, key=lambda p: fp[p]["region_iou"] - base[p]["region_iou"])[:5]
    print("  worst regressions:", [(p, round(base[p]["region_iou"], 2), round(fp[p]["region_iou"], 2), round(fp[p]["tilt"])) for p in worst])
    best = sorted(rect, key=lambda p: base[p]["region_iou"] - fp[p]["region_iou"])[:5]
    print("  best gains:", [(p, round(base[p]["region_iou"], 2), round(fp[p]["region_iou"], 2), round(fp[p]["tilt"])) for p in best])


if __name__ == "__main__":
    main()
