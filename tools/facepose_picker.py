import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(fp_name="gate_bench_facepose2.json"):
    base = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "runs", "gate_bench_base.json"))) if "region_iou" in r}
    fp = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "runs", fp_name))) if "region_iou" in r}
    parts = sorted(set(base) & set(fp))
    rect = [p for p in parts if fp[p].get("rectified") and fp[p].get("agreement")]
    b = np.array([base[p]["region_iou"] for p in rect]); f = np.array([fp[p]["region_iou"] for p in rect])
    agree = np.array([fp[p]["agreement"][fp[p]["view"]] for p in rect])
    inl = np.array([fp[p]["face_inliers"] if fp[p].get("face_inliers") is not None else 0 for p in rect])
    tilt = np.array([fp[p]["tilt"] for p in rect])
    n_all = len(parts); base_all = np.mean([base[p]["region_iou"] for p in parts])
    print("PICKER rectified=%d of %d | base mean (rectified parts) %.3f | always-rectify %.3f | oracle-per-part %.3f | base all %.3f"
          % (len(rect), n_all, b.mean(), f.mean(), np.maximum(b, f).mean(), base_all))
    def report(name, use):
        pick = np.where(use, f, b)
        print("  %-38s n_use=%2d  mean %.3f (delta %+.3f)  wins %2d losses %2d" % (name, int(use.sum()), pick.mean(), pick.mean() - b.mean(),
              int(((f - b) > 0.01)[use].sum()), int(((f - b) < -0.01)[use].sum())))
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        report("agreement > %.1f" % t, agree > t)
    for t in (0.5, 0.7, 0.8, 0.9):
        report("face_inliers > %.1f" % t, inl > t)
    for lo, hi in ((25, 90), (30, 90), (30, 45), (35, 60)):
        report("tilt in [%d,%d)" % (lo, hi), (tilt >= lo) & (tilt < hi))
    for ta, tt in ((0.7, 30), (0.8, 30), (0.7, 25), (0.6, 30)):
        report("agreement > %.1f and tilt >= %d" % (ta, tt), (agree > ta) & (tilt >= tt))
    report("agreement > 0.7 and face_inliers > 0.7", (agree > 0.7) & (inl > 0.7))
    from scipy.stats import spearmanr
    print("  spearman(delta, agreement) %.2f  spearman(delta, face_inliers) %.2f  spearman(delta, tilt) %.2f"
          % (spearmanr(f - b, agree).correlation, spearmanr(f - b, inl).correlation, spearmanr(f - b, tilt).correlation))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gate_bench_facepose2.json")
