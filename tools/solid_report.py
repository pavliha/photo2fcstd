import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(path=os.path.join(ROOT, "runs", "solid_bench.json"), base=None):
    rows = json.load(open(path))
    fam = json.load(open(os.path.join(ROOT, "runs", "template_families.json"))); inv = {p: k for k, ps in fam.items() for p in ps}
    ok = [r for r in rows if "iou3d" in r]; ref = [r for r in rows if "refused" in r]; err = [r for r in rows if "error" in r]
    iou = np.array([r["iou3d"] for r in ok])
    print("%s: n=%d built=%d refused=%d errors=%d | 3D IoU mean %.3f median %.3f | >=0.8 %d (%.0f%% of all)  >=0.6 %d  <0.3 %d"
          % (os.path.basename(path), len(rows), len(ok), len(ref), len(err), iou.mean(), np.median(iou), (iou >= 0.8).sum(), 100 * (iou >= 0.8).sum() / len(rows), (iou >= 0.6).sum(), (iou < 0.3).sum()))
    fams = {}
    for r in ok:
        fams.setdefault(inv.get(r["part"], "?"), []).append(r["iou3d"])
    for k, v in sorted(fams.items(), key=lambda kv: -len(kv[1]))[:10]:
        print("  %-20s n=%3d  3D IoU %.3f  >=0.8 %d" % (k, len(v), np.mean(v), sum(x >= 0.8 for x in v)))
    print("  refused reasons:", {})
    reasons = {}
    for r in ref:
        reasons[r["refused"][:38]] = reasons.get(r["refused"][:38], 0) + 1
    print("  ", sorted(reasons.items(), key=lambda kv: -kv[1])[:5])
    worst = sorted(ok, key=lambda r: r["iou3d"])[:12]
    print("  worst:", [(r["part"], inv.get(r["part"]), r.get("mode"), round(r["iou3d"], 2)) for r in worst])
    if base:
        b = {r["part"]: r for r in json.load(open(base))}
        common = [r["part"] for r in ok if r["part"] in b and "iou3d" in b[r["part"]]]
        a1 = np.array([b[p]["iou3d"] for p in common]); a2 = np.array([[r for r in ok if r["part"] == p][0]["iou3d"] for p in common])
        print("  paired vs %s: n=%d  %.3f -> %.3f  wins %d losses %d" % (os.path.basename(base), len(common), a1.mean(), a2.mean(), (a2 > a1 + 0.05).sum(), (a2 < a1 - 0.05).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:])
