"""Does the depth model survive a dataset it was not trained on?

It predicts log(depth / length) from silhouette statistics and was fitted entirely on PrintCAD,
the same single-dataset provenance that left the axis classifier at chance on T-LESS. T-LESS has
real photographs and a CAD mesh, so the true ratio is known and the claim is checkable.
"""
import json, os, sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2fcstd import analysis, depth_model, tless  # noqa: E402
from tless_sketch import reference_axis  # noqa: E402


def truth_ratio(mesh, axis):
    ext = mesh.extents
    face = [ext[a] for a in (0, 1, 2) if a != axis]
    return float(ext[axis] / max(max(face), 1e-9))


def main(limit=30):
    rows = []
    for obj in range(1, limit + 1):
        try:
            mesh = trimesh.load(tless.model_path(obj))
            ax, var = reference_axis(mesh)
            if ax is None:
                continue
            pairs = tless.views_of(obj, every=90, limit=6)
            masks = [m for m in tless.masks_of(pairs) if m is not None and m.sum() > 400]
            if len(masks) < 3:
                continue
            views = []
            for m in masks[:3]:
                try:
                    views.append(analysis.view_from_mask(m, "obj%02d" % obj))
                except Exception:
                    pass
            if len(views) < 3:
                continue
            got = depth_model.predict(views)
            if got is None:
                continue
            rows.append({"obj": obj, "pred": got[0], "true": truth_ratio(mesh, ax),
                         "lo": got[1], "hi": got[2]})
        except Exception:
            continue
    json.dump(rows, open(os.path.join(ROOT, "data", "depth_ood.json"), "w"))
    if not rows:
        print("no T-LESS objects produced both a prediction and a truth ratio")
        return
    p = np.array([r["pred"] for r in rows])
    t = np.array([r["true"] for r in rows])
    err = np.abs(np.log(p / t))
    best_const = float(np.exp(np.median(np.log(t))))
    const_err = np.abs(np.log(best_const / t))
    print("depth ratio on T-LESS, n=%d objects\n" % len(rows))
    print("  %-30s %10s %10s" % ("", "median |log|", "within 2x"))
    print("  %-30s %10.3f %9.0f%%" % ("best constant, fitted on T-LESS", np.median(const_err),
                                      100 * np.mean(const_err < np.log(2))))
    print("  %-30s %10.3f %9.0f%%" % ("learned model", np.median(err), 100 * np.mean(err < np.log(2))))
    print("\n  on PrintCAD it was 0.435 against a constant's 1.045")
    band = [r for r in rows if r["lo"] and r["hi"]]
    if band:
        cov = np.mean([r["lo"] <= r["true"] <= r["hi"] for r in band])
        width = np.median([r["hi"] / r["lo"] for r in band])
        print("  band covers the truth %.0f%% of the time (claims 80%%), median width %.1fx"
              % (100 * cov, width))
    print("\n  true ratio spans %.3f to %.3f; the model predicts %.3f to %.3f"
          % (t.min(), t.max(), p.min(), p.max()))


if __name__ == "__main__":
    main()
