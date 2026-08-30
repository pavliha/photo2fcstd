import argparse
import json
import os
import sys

import numpy as np

from photo2fcstd import telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODES = ("stations", "profile", "plan", "revolve")


def results(run):
    path = os.path.join(ROOT, "runs", run, "results.txt")
    rows = [l.split() for l in open(path) if len(l.split()) == 3]
    return {k: (m, float(v)) for k, m, v in rows if v != "fail"}


def forced(prefix="forced_"):
    return {m: results(prefix + m) for m in MODES if os.path.exists(os.path.join(ROOT, "runs", prefix + m, "results.txt"))}


def oracle(per_mode):
    parts = set.union(*[set(r) for r in per_mode.values()])
    out = {}
    for p in parts:
        scored = [(r[p][1], m) for m, r in per_mode.items() if p in r]
        best, mode = max(scored)
        out[p] = {"best_mode": mode, "best_iou": best, "per_mode": {m: r[p][1] for m, r in per_mode.items() if p in r}}
    return out


def report(run, per_mode, events_path=None):
    actual = results(run)
    ora = oracle(per_mode)
    common = [p for p in actual if p in ora]
    a = np.array([actual[p][1] for p in common])
    o = np.array([ora[p]["best_iou"] for p in common])
    print("run %s: n=%d  mean %.3f   oracle mode %.3f   gap %.3f" % (run, len(common), a.mean(), o.mean(), o.mean() - a.mean()))
    right = [p for p in common if actual[p][0] == ora[p]["best_mode"]]
    print("  mode chosen best for %d of %d parts (%.0f%%)" % (len(right), len(common), 100 * len(right) / len(common)))
    print("  confusion (chosen -> best):")
    for chosen in MODES:
        picks = [p for p in common if actual[p][0] == chosen]
        if not picks:
            continue
        counts = {}
        for p in picks:
            counts[ora[p]["best_mode"]] = counts.get(ora[p]["best_mode"], 0) + 1
        loss = sum(ora[p]["best_iou"] - actual[p][1] for p in picks) / len(common)
        print("    %-9s n=%3d  %s   gap contribution %.3f" % (chosen, len(picks), counts, loss))
    worst = sorted(common, key=lambda p: actual[p][1] - ora[p]["best_iou"])[:12]
    print("  biggest mode-selection losses:")
    for p in worst:
        print("    %s chose %-9s %.2f, best %-9s %.2f   %s" % (p, actual[p][0], actual[p][1], ora[p]["best_mode"], ora[p]["best_iou"],
                                                               {m: round(v, 2) for m, v in ora[p]["per_mode"].items()}))
    if events_path and os.path.exists(events_path):
        feature_report(events_path, actual, ora)
    return {"mean": float(a.mean()), "oracle": float(o.mean()), "mode_accuracy": len(right) / len(common)}


def feature_report(events_path, actual, ora):
    events = {e["part"]: e for e in telemetry.load(events_path)}
    common = [p for p in actual if p in events and "views" in events[p]]
    print("  feature correlations with IoU (n=%d):" % len(common))
    iou = np.array([actual[p][1] for p in common])
    features = {
        "min elongation": lambda e: min(v["elongation"] for v in e["views"]),
        "max rectangularity": lambda e: max(v["rectangularity"] for v in e["views"]),
        "min min_over_max": lambda e: min(v["min_over_max"] for v in e["views"]),
        "max hole_frac": lambda e: max(v["hole_frac"] for v in e["views"]),
        "min ellipse_rms": lambda e: min(v["ellipse_rms"] for v in e["views"]),
        "primitives": lambda e: sum(e["spec"]["primitives"].values()) if "spec" in e else 0,
        "stations": lambda e: max(e["spec"]["station_counts"].values()) if e.get("spec", {}).get("station_counts") else 0,
    }
    for name, fn in features.items():
        x = np.array([fn(events[p]) for p in common], float)
        if x.std() > 0:
            print("    %-20s r=%+.2f" % (name, np.corrcoef(x, iou)[0, 1]))
    unconstrained = [p for p in common if events[p].get("build", {}).get("fully_constrained") is False]
    print("  sketches not fully constrained: %d" % len(unconstrained))


def dataset(per_mode, events_path, out):
    events = {e["part"]: e for e in telemetry.load(events_path)}
    ora = oracle(per_mode)
    rows = [{"part": p, "views": events[p]["views"], "label": ora[p]["best_mode"],
             "iou_per_mode": ora[p]["per_mode"]} for p in sorted(ora) if p in events and "views" in events[p]]
    json.dump(rows, open(out, "w"))
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("wrote %s: %d parts, labels %s" % (out, len(rows), counts))
    return rows


def label_quality(rows, ambiguous=0.05, floor=0.2):
    import numpy as np
    per = [r["iou_per_mode"] for r in rows]
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    modes_present = np.array([len(d) for d in per])
    best = np.array([max(d.values()) for d in per])
    second = np.array([sorted(d.values())[-2] if len(d) > 1 else 0.0 for d in per])
    margin = best - second
    unlearnable = best < floor
    ties = (~unlearnable) & (margin < ambiguous)
    clear = (~unlearnable) & (margin >= ambiguous)
    print("labels: %d parts   %s" % (len(rows), counts))
    print("  modes that built per part: mean %.2f of 4 (all four for %d parts)" % (modes_present.mean(), int((modes_present == 4).sum())))
    print("  best-mode IoU: median %.2f, quartiles %.2f/%.2f" % (np.median(best), *np.percentile(best, [25, 75])))
    print("  margin over second best: median %.3f, quartiles %.3f/%.3f" % (np.median(margin), *np.percentile(margin, [25, 75])))
    print("  unlearnable (best < %.2f, no mode works): %d (%.0f%%)" % (floor, unlearnable.sum(), 100 * unlearnable.mean()))
    print("  ambiguous (margin < %.2f, label is a coin flip): %d (%.0f%%)" % (ambiguous, ties.sum(), 100 * ties.mean()))
    print("  decisive labels worth learning: %d (%.0f%%)" % (clear.sum(), 100 * clear.mean()))
    reachable = float(np.mean([b if c else s for b, s, c in zip(best, second, clear)]))
    print("  ceiling if every decisive label were predicted: %.3f (oracle %.3f)" % (reachable, best.mean()))
    return {"n": len(rows), "decisive": int(clear.sum()), "unlearnable": int(unlearnable.sum()), "ambiguous": int(ties.sum())}


def main(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd-insight")
    p.add_argument("run")
    p.add_argument("--forced-prefix", default="forced_")
    p.add_argument("--dataset", help="write a mode-labelled training set here")
    p.add_argument("--quality", help="report the quality of an existing label set")
    a = p.parse_args(argv)
    if a.quality:
        label_quality(json.load(open(a.quality)))
        return
    per_mode = forced(a.forced_prefix)
    if not per_mode:
        raise SystemExit("no forced runs found: photo2fcstd-bench forced_<mode> --mode <mode>")
    events_path = os.path.join(ROOT, "runs", a.run, "events.jsonl")
    report(a.run, per_mode, events_path)
    if a.dataset:
        dataset(per_mode, events_path, a.dataset)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
