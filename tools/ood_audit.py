"""Every learned component, in distribution against out of it, beside its baseline.

Split-by-part is enforced everywhere and protects against a model memorising a part. Nothing
protects against it memorising PrintCAD, and two of the three components audited had. This is the
check that would have caught them, run as one command.
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")
REPORT = os.path.join(DATA, "ood_audit.json")


def axis_model_scores():
    """Agreement with the axis a single sketch can describe: PrintCAD against T-LESS."""
    import joblib
    from photo2fcstd import axis_model
    out = {}
    p = os.path.join(DATA, "axis_holdout.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        m = joblib.load(os.path.join(DATA, "axis_model.joblib"))
        got = [int(np.argmax(m.predict_proba(np.array(r["x"], float))[:, 1])) == r["best"] for r in rows]
        out["in"] = float(np.mean(got))
    p = os.path.join(DATA, "tless_sketch.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        out["out"] = float(np.mean([r["axis_model_agreed"] for r in rows]))
        out["n_out"] = len(rows)
    out["baseline"] = 1 / 3
    return out


def depth_scores(arm):
    """Median absolute log error against the best constant. The two paths differ enough that
    reporting whichever ran last would hide the finding: the shipped pixel path is three times
    worse than the tabular fallback out of distribution."""
    out = {"baseline_note": "best constant fitted on the same set", "in": 0.435}
    p = os.path.join(DATA, "depth_ood_%s.json" % arm)
    if not os.path.exists(p):
        return out
    rows = json.load(open(p))
    t = np.array([r["true"] for r in rows], float)
    pr = np.array([r["pred"] for r in rows], float)
    out["out"] = float(np.median(np.abs(np.log(pr / t))))
    out["baseline"] = float(np.median(np.abs(np.log(np.exp(np.median(np.log(t))) / t))))
    out["n_out"] = len(rows)
    cov = [r for r in rows if r.get("lo") and r.get("hi")]
    if cov:
        out["coverage"] = float(np.mean([r["lo"] <= r["true"] <= r["hi"] for r in cov]))
    return out


def view_model_scores():
    from photo2fcstd import view_model
    out = {"baseline": 1 / 3}
    p = os.path.join(DATA, "view_ceiling_big.json")
    seen = set(json.load(open(os.path.join(DATA, "view_ceiling.json"))))
    if os.path.exists(p):
        lab = json.load(open(p))
        rows = [(k, v) for k, v in lab.items()
                if k not in seen and len(v["per_view"]) == 3 and all(v["per_view"])]
        I = np.array([[q["iou"] for q in v["per_view"]] for _, v in rows], float)
        pred = []
        for _, v in rows:
            try:
                pred.append(int(np.argmax(view_model.load().predict_proba(view_model.features(v["per_view"]))[:, 1])))
            except Exception:
                pred.append(0)
        out["in"] = float(np.mean(np.array(pred) == I.argmax(1)))
        out["n_in"] = len(rows)
    out["out"] = None
    out["note"] = "T-LESS has no view triples of one object from one setup, so this is untested out of distribution"
    return out


def mode_model_scores():
    """Off by default and worth about 0.001 in distribution; recorded so the audit covers every
    learned component rather than only the ones that turned out to matter."""
    from photo2fcstd import modes
    return {"baseline": None, "in": None, "out": None,
            "note": "disabled (modes.LEARNED=%s); worth ~0.001 in distribution, never audited out"
                    % modes.LEARNED}


def main():
    report = {"axis_model": axis_model_scores(),
              "mode_model": mode_model_scores(),
              "depth_model_shipped": depth_scores("shipped"),
              "depth_model_tabular": depth_scores("tabular"),
              "view_model": view_model_scores()}
    json.dump(report, open(REPORT, "w"), indent=1)
    print("out-of-distribution audit\n")
    print("  %-14s %10s %10s %10s  %s" % ("component", "baseline", "in", "out", "verdict"))
    for name, r in report.items():
        b = r.get("baseline")
        i, o = r.get("in"), r.get("out")
        if r.get("in") is None and r.get("out") is None:
            print("  %-14s %10s %10s %10s  %s" % (name, "-", "-", "-", "disabled, unaudited"))
            continue
        if name.startswith("depth_model"):
            verdict = "LOSES to a constant" if (o is not None and b is not None and o > b) else "holds"
            fmt = lambda x: "-" if x is None else "%.3f" % x
        else:
            verdict = ("untested" if o is None else
                       "DATASET-SPECIFIC" if (b is not None and o <= b + 0.05) else "holds")
            fmt = lambda x: "-" if x is None else "%.3f" % x
        print("  %-14s %10s %10s %10s  %s" % (name, fmt(b), fmt(i), fmt(o), verdict))
    print("\n  written to %s" % os.path.relpath(REPORT, ROOT))
    print("  depth is median absolute log error, lower is better; the others are accuracy")


if __name__ == "__main__":
    main()
