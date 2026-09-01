"""Two learned view selectors exist for one decision. Measure both, keep one."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import stats, view_model, view_rank  # noqa: E402

lab = json.load(open(os.path.join(ROOT, "data", "view_ceiling_big.json")))
seen_by_view_model = set(json.load(open(os.path.join(ROOT, "data", "view_ceiling.json"))))
rows = [(k, v) for k, v in lab.items()
        if k not in seen_by_view_model and len(v["per_view"]) == 3 and all(v["per_view"])]
I = np.array([[p["iou"] for p in v["per_view"]] for _, v in rows], float)
truth = I.argmax(1)
idx = np.arange(len(rows))


def fake_views(per_view):
    """view_rank and view_model read different fields; give each what it expects."""
    out = []
    for p in per_view:
        out.append({"shape": {"bbox": (np.sqrt(max(p["area"], 1.0)), np.sqrt(max(p["area"], 1.0))),
                              "rectangularity": p["rect"], "solidity": p["sol"],
                              "hole_frac": p["hole_frac"], "ellipse_rms": p["ellipse_rms"],
                              "stroke_px": p["stroke"], "holes": [0] * int(p["nholes"])},
                    "elongation": p["elong"], "symmetric": [], "length_px": max(p["area"], 1.0)})
    return out


def pick(fn, name):
    got = []
    for _, v in rows:
        try:
            got.append(fn(v))
        except Exception:
            got.append(0)
    c = np.array(got, int)
    print("  %-22s agree %.2f   IoU %.3f" % (name, np.mean(c == truth), I[idx, c].mean()))
    return I[idx, c]


def by_view_model(v):
    i = view_model.choose(fake_views(v["per_view"]))
    return 0 if i is None else i


def by_ranker(v):
    vs = fake_views(v["per_view"])
    best = view_rank.best(vs)
    return vs.index(best) if best is not None else 0


print("n=%d parts, none seen by view_model in training\n" % len(rows))
print("  %-22s agree %.2f   IoU %.3f" % ("chance", 1 / 3, I.mean()))
print("  %-22s %s   IoU %.3f" % ("first photo", "     ", I[:, 0].mean()))
a = pick(by_view_model, "view_model")
b = pick(by_ranker, "view_rank")
print("  %-22s agree %.2f   IoU %.3f" % ("oracle", 1.0, I.max(1).mean()))
d = b - a
m, lo, hi = stats.mean_ci(d)
print("\n  view_rank minus view_model: %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
