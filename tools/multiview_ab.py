import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(pattern=os.path.join(ROOT, "runs", "multiview_bench.json")):
    import glob
    rows = [r for f in sorted(glob.glob(pattern)) for r in json.load(open(f)) if "region_iou_mv" in r]
    base = np.array([r["region_iou"] for r in rows]); mv = np.array([r["region_iou_mv"] for r in rows])
    fit = np.array([min(r["fit_iou"]) for r in rows]); tilt = np.array([r["tilts"][r["view"]] for r in rows])
    gain = mv - base
    print("fitted n=%d  base %.3f  mv %.3f  wins %d losses %d" % (len(rows), base.mean(), mv.mean(), (gain > 0.02).sum(), (gain < -0.02).sum()))
    print("oracle %.3f" % np.maximum(base, mv).mean())
    for fmin in (0.0, 0.85, 0.9, 0.93, 0.95):
        for tmin in (0, 8, 15, 25):
            use = (fit >= fmin) & (tilt >= tmin)
            pick = np.where(use, mv, base)
            print("fit>=%.2f tilt>=%2d  n=%3d  picked %.3f  (%+.3f)  wins %d losses %d"
                  % (fmin, tmin, use.sum(), pick.mean(), pick.mean() - base.mean(), (gain[use] > 0.02).sum(), (gain[use] < -0.02).sum()))
    order = np.argsort(gain)
    print("worst:", [(rows[i]["part"], round(float(gain[i]), 3), round(float(fit[i]), 2), round(float(tilt[i]))) for i in order[:6]])
    print("best: ", [(rows[i]["part"], round(float(gain[i]), 3), round(float(fit[i]), 2), round(float(tilt[i]))) for i in order[::-1][:6]])


if __name__ == "__main__":
    main(*sys.argv[1:])
