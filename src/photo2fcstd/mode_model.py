import argparse
import json
import os
import sys

import numpy as np

MODES = ("stations", "profile", "plan", "revolve")
VIEW_FIELDS = ("elongation", "rectangularity", "solidity", "hole_frac", "min_over_max",
               "ellipse_rms", "stroke_px", "stations")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("P2F_MODE_MODEL", os.path.join(ROOT, "data", "mode_model.joblib"))


def features(views):
    ordered = sorted(views, key=lambda v: -v["elongation"])[:3]
    row = []
    for v in ordered:
        row += [float(v[f]) for f in VIEW_FIELDS]
        row += [float(v["ellipse_aspect"] or 0.0), float(v["round"]), float(v["roundish"]),
                float(v["stroke_px"]) / max(v["length_px"], 1.0), float(len(v["symmetric"]))]
    while len(row) < 3 * (len(VIEW_FIELDS) + 5):
        row += row[:len(VIEW_FIELDS) + 5]
    rect = [v["rectangularity"] for v in ordered]
    elong = [v["elongation"] for v in ordered]
    sol = [v["solidity"] for v in ordered]
    row += [max(rect) - min(rect), max(sol) - min(sol), max(elong) / max(min(elong), 1e-6),
            max(v["hole_frac"] for v in ordered), sum(v["round"] for v in ordered), len(ordered)]
    return row


def matrix(rows):
    return np.array([features(r["views"]) for r in rows], float)


def iou_of(row, mode):
    return row["iou_per_mode"].get(mode, 0.0)


def evaluate(rows, predicted):
    return float(np.mean([iou_of(r, m) for r, m in zip(rows, predicted)]))


def cross_validated(rows, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    X = matrix(rows)
    y = np.array([r["label"] for r in rows])
    predicted = np.empty(len(rows), dtype=object)
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=seed, class_weight="balanced")
        model.fit(X[train], y[train])
        predicted[test] = model.predict(X[test])
    return list(predicted)


def outcome_targets(rows):
    return np.array([[r["iou_per_mode"].get(m, np.nan) for m in MODES] for r in rows], float)


def cross_validated_regression(rows, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    X, Y = matrix(rows), outcome_targets(rows)
    predicted = np.empty(len(rows), dtype=object)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(X):
        heads = {}
        for j, mode in enumerate(MODES):
            usable = train[np.isfinite(Y[train, j])]
            if len(usable) < 10:
                continue
            head = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed)
            head.fit(X[usable], Y[usable, j])
            heads[mode] = head
        for i in test:
            buildable = [m for j, m in enumerate(MODES) if m in heads and np.isfinite(Y[i, j])]
            if not buildable:
                predicted[i] = rows[i]["label"]
                continue
            predicted[i] = max(buildable, key=lambda m: heads[m].predict(X[i:i + 1])[0])
    return list(predicted)


def train(rows, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=seed, class_weight="balanced")
    model.fit(matrix(rows), [r["label"] for r in rows])
    return {"kind": "classifier", "model": model}


def train_regression(rows, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    X, Y = matrix(rows), outcome_targets(rows)
    heads = {}
    for j, mode in enumerate(MODES):
        usable = np.isfinite(Y[:, j])
        if usable.sum() < 10:
            continue
        head = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed)
        head.fit(X[usable], Y[usable, j])
        heads[mode] = head
    return {"kind": "regression", "heads": heads}


_CACHED = {}


def predict(views, allowed=None):
    import joblib
    from photo2fcstd import fallback
    if not os.path.exists(MODEL_PATH):
        fallback.note("mode_model", "no model at %s" % MODEL_PATH)
        return None
    if MODEL_PATH not in _CACHED:
        _CACHED[MODEL_PATH] = joblib.load(MODEL_PATH)
    saved = _CACHED[MODEL_PATH]
    x = np.array([features(views)], float)
    if isinstance(saved, dict) and saved.get("kind") == "regression":
        heads = {m: h for m, h in saved["heads"].items() if allowed is None or m in allowed}
        if not heads:
            fallback.note("mode_model", "no head for any allowed mode")
            return None
        return max(heads, key=lambda m: heads[m].predict(x)[0])
    model = saved["model"] if isinstance(saved, dict) else saved
    if allowed is None:
        return str(model.predict(x)[0])
    classes = [str(c) for c in getattr(model, "classes_", [])]
    keep = [i for i, c in enumerate(classes) if c in allowed]
    if not keep:
        fallback.note("mode_model", "trained on %s, none of which is allowed here" % classes)
        return None
    try:
        probs = model.predict_proba(x)[0]
    except Exception as exc:
        fallback.note("mode_model", "predict_proba failed: %s" % exc)
        return None
    return classes[max(keep, key=lambda i: probs[i])]


def main(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd-train-modes")
    p.add_argument("dataset", nargs="?", default=os.path.join(ROOT, "data", "mode_labels.json"))
    p.add_argument("--out", default=MODEL_PATH)
    p.add_argument("--rules-run", default="v12")
    p.add_argument("--regression", action="store_true", help="save per-mode outcome heads instead of a classifier")
    a = p.parse_args(argv)
    rows = json.load(open(a.dataset))
    oracle = float(np.mean([r["iou_per_mode"][r["label"]] for r in rows]))
    cv = cross_validated(rows)
    learned = evaluate(rows, cv)
    accuracy = float(np.mean([m == r["label"] for r, m in zip(rows, cv)]))
    reg = cross_validated_regression(rows)
    reg_iou = evaluate(rows, reg)
    reg_acc = float(np.mean([m == r["label"] for r, m in zip(rows, reg)]))
    from photo2fcstd.insight import results
    rules = results(a.rules_run)
    rule_iou = float(np.mean([iou_of(r, rules[r["part"]][0]) for r in rows if r["part"] in rules]))
    rule_acc = float(np.mean([rules[r["part"]][0] == r["label"] for r in rows if r["part"] in rules]))
    print("mode selection on %d parts" % len(rows))
    print("  rules      accuracy %.0f%%   mean IoU %.3f" % (100 * rule_acc, rule_iou))
    print("  learned CV accuracy %.0f%%   mean IoU %.3f" % (100 * accuracy, learned))
    print("  per-mode regression, argmax: accuracy %.0f%%   mean IoU %.3f" % (100 * reg_acc, reg_iou))
    print("  oracle                        mean IoU %.3f" % oracle)
    import joblib
    joblib.dump(train_regression(rows) if a.regression else train(rows), a.out)
    print("  wrote %s (%s)" % (a.out, "per-mode regression" if a.regression else "classifier"))
    model = train(rows)["model"]
    names = [f + "_v%d" % i for i in range(3) for f in list(VIEW_FIELDS) + ["ellipse_aspect", "round", "roundish", "stroke_frac", "symmetry"]] + \
            ["rect_spread", "solidity_spread", "elongation_ratio", "max_hole_frac", "round_views", "n_views"]
    top = sorted(zip(model.feature_importances_, names), reverse=True)[:8]
    print("  top features: %s" % ", ".join("%s %.2f" % (n, v) for v, n in top))


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
