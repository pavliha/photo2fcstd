"""Re-judge the shipped and reverted A/Bs on structure rather than on region IoU alone.

The dumps kept different fields, so each pair is judged on whatever structural evidence it
recorded. Where a pair kept only `exact`, that is the whole structural signal available and the
comparison is weaker; that is stated rather than papered over.
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import stats  # noqa: E402

PAIRS = [
    ("view choice", "ab_view_shipped.json", "ab_view_view_model.json", "shipped"),
    ("mode allowed-list", "ab_modefix_before.json", "ab_modefix_fixed.json", "shipped"),
    ("merge runs into arcs", "ab_merge_shipped.json", "ab_merge_merge.json", "reverted"),
]


def load(name):
    p = os.path.join(ROOT, "data", name)
    return json.load(open(p)) if os.path.exists(p) else None


def structure(rec):
    """The terms this dump can support, each a fraction of the real sketch reproduced."""
    terms = {"exact": float(bool(rec.get("exact")))}
    if "n" in rec and "ideal_n" in rec:
        terms["elements"] = max(1.0 - abs(rec["n"] - rec["ideal_n"]) / max(rec["ideal_n"], 1), 0.0)
    if "curve" in rec and "curve_ideal" in rec:
        terms["curves"] = 1.0 - abs(rec["curve"] - rec["curve_ideal"])
    return terms


def main():
    for name, a_f, b_f, was in PAIRS:
        A, B = load(a_f), load(b_f)
        if not A or not B:
            print("  %-22s dumps not kept" % name)
            continue
        keys = sorted(k for k in A if "iou" in A.get(k, {}) and "iou" in B.get(k, {})
                      and 1 - A[k].get("trivial", 0) >= 0.15)
        d = np.array([B[k]["iou"] - A[k]["iou"] for k in keys])
        m, lo, hi = stats.mean_ci(d)
        print("  %s  (n=%d, %s)" % (name, len(keys), was))
        print("      region IoU     %+.4f  [%+.4f, %+.4f]" % (m, lo, hi))
        for term in ("exact", "elements", "curves"):
            va = [structure(A[k]).get(term) for k in keys]
            if any(v is None for v in va):
                continue
            vb = [structure(B[k])[term] for k in keys]
            dm, dlo, dhi = stats.mean_ci(np.array(vb) - np.array(va))
            flag = "" if (dm > 0) == (m > 0) or abs(dm) < 1e-9 else "   <- disagrees with IoU"
            print("      %-13s %+.4f  [%+.4f, %+.4f]%s" % (term, dm, dlo, dhi, flag))
        print()


if __name__ == "__main__":
    main()
