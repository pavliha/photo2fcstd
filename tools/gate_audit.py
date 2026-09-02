"""Does each gate's own score point the right way?

`axis_model`'s confidence runs backwards out of distribution - it scores 0.73 when wrong against
0.56 when right - so gating on it abstains precisely on the cases it gets right. A gate reading a
backwards signal is worse than no gate, so every gate needs the same check: does the quantity it
thresholds actually rise with the error it is meant to catch?
"""
import json, os, sys

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def depth_gate():
    """embedding_is_familiar: cosine distance from the training centre, thresholded at the 95th pct."""
    from depth_pixel_train import data, pipeline
    from photo2fcstd import depth_model as DM
    X, y, g = data(hybrid=False)
    ref = dict(np.load(DM.CENTRE_PATH))
    centre = ref[[k for k in ref if "centre" in k or k == "mu"][0]] if any("centre" in k or k == "mu" for k in ref) \
        else list(ref.values())[0]
    centre = np.asarray(ref["centre"], float).ravel()
    per = X.reshape(X.shape[0], -1, centre.shape[0])
    per = per / np.maximum(np.linalg.norm(per, axis=2, keepdims=True), 1e-9)
    dist = np.median(1.0 - per @ centre, axis=1)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = pipeline().fit(X[tr], y[tr]).predict(X[te])
    err = np.abs(pred - y)
    return "depth: embedding_is_familiar", float(np.corrcoef(dist, err)[0, 1]), "%d parts" % len(y)


def tilt_gate():
    """tilt_model.is_familiar: same construction, on the tilt embeddings."""
    import joblib
    from sklearn.linear_model import RidgeCV
    path = os.path.join(ROOT, "data", "tilt_model.joblib")
    if not os.path.exists(path):
        return "tilt: is_familiar", None, "no artefact"
    m = joblib.load(path)
    d = np.load(os.path.join(ROOT, "data", "tilt_pixels.npz"))
    X, Y, g = d["obj"], d["normal"], d["part"]
    dist = 1.0 - X @ m["centre"]
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        for j in range(3):
            pred[te, j] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(
                (X[tr] - mu) / sd, Y[tr, j]).predict((X[te] - mu) / sd)
    p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    err = np.degrees(np.arccos(np.clip(np.abs(np.sum(p * Y, axis=1)), -1, 1)))
    return "tilt: is_familiar", float(np.corrcoef(dist, err)[0, 1]), "%d views" % len(Y)


def main():
    print("  a gate is sound when its score rises with the error it exists to catch,\n"
          "  so a POSITIVE correlation is right and a negative one is backwards\n")
    print("  %-32s %12s   %s" % ("gate", "correlation", "n"))
    for fn in (depth_gate, tilt_gate):
        try:
            name, corr, note = fn()
        except Exception as exc:
            print("  %-32s %12s   %s" % (fn.__name__, "failed", str(exc)[:60]))
            continue
        verdict = "not measured" if corr is None else (
            "sound" if corr > 0.1 else ("BACKWARDS" if corr < -0.1 else "no signal either way"))
        print("  %-32s %12s   %s   %s" % (name, "-" if corr is None else "%+.2f" % corr, note, verdict))
    print("\n  recorded for comparison: axis_model scores 0.73 when wrong and 0.56 when right\n"
          "  out of distribution, which is the backwards case this check exists for.")


if __name__ == "__main__":
    main()
