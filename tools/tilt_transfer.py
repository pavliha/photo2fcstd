"""Does the T-LESS tilt head survive PrintCAD? Two questions, because they fail differently.

1. **Does the gate admit a real PrintCAD photograph?** Measurable on real images with no tilt
   truth needed, and it decides whether the head may speak at all.
2. **Is it accurate on PrintCAD geometry?** Only answerable on synthetic renders, since no
   PrintCAD photo has a known tilt - so a failure here confounds "different parts" with "not a
   photograph", and is reported as such rather than as a clean transfer number.
"""
import json, os, sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from PIL import Image  # noqa: E402
from photo2fcstd import bench, embed, sketch_score as SS  # noqa: E402


def predict(model, vectors):
    x = (vectors - model["mu"]) / model["sd"]
    p = np.stack([h.predict(x) for h in model["heads"]], axis=1)
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)


def gate_on_real_photos(model, limit=60):
    parts = bench.with_photos([p for p in sorted(SS_ideal())])[0][:limit]
    paths = [bench.photos_of(p)[0] for p in parts]
    v = embed.vectors_for(paths)
    dist = 1.0 - v @ model["centre"]
    inside = float(np.mean(dist <= model["gate"]))
    print("  real PrintCAD photographs, n=%d" % len(paths))
    print("    cosine distance to the T-LESS training centre: median %.3f, gate at %.3f"
          % (np.median(dist), model["gate"]))
    print("    admitted by the gate: %.0f%%\n" % (100 * inside))
    return dist


def SS_ideal():
    return json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def accuracy_on_renders(model, limit=80, tilts=(0.0, 8.0, 15.0, 25.0)):
    import tilt_close as T
    ideal = T.IDEAL
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p]) and T.face_normal(ideal[p]) is not None][:limit]
    imgs, truth, dists = [], [], []
    for part in parts:
        n = T.face_normal(ideal[part])
        try:
            m = trimesh.load(bench.truth_of(part))
            assert len(m.faces)
        except Exception:
            continue
        m.apply_translation(-m.bounds.mean(0))
        rad = float(np.max(m.extents)) * 4
        for t in tilts:
            v = T.view_at(n, t, 40.0, rad)
            img, mask = T.shaded(m, v)
            if mask.sum() < 200:
                continue
            ys, xs = np.nonzero(mask)
            pad = int(0.15 * max(np.ptp(xs), np.ptp(ys)))
            box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
                   min(T.W, xs.max() + pad), min(T.H, ys.max() + pad))
            rgb = np.dstack([img * mask] * 3)[box[1]:box[3], box[0]:box[2]]
            imgs.append(Image.fromarray(rgb.astype(np.uint8)).resize((embed.SIZE, embed.SIZE)))
            truth.append(v["R"] @ n)
    vecs = np.concatenate([embed.embed_images(imgs[i:i + 16]) for i in range(0, len(imgs), 16)])
    Y = np.array(truth)
    Y *= np.sign(np.where(Y[:, 2:3] == 0, 1, Y[:, 2:3]))
    p = predict(model, vecs)
    err = np.degrees(np.arccos(np.clip(np.abs(np.sum(p * Y, axis=1)), -1, 1)))
    dist = 1.0 - vecs @ model["centre"]
    print("  synthetic PrintCAD renders, n=%d over %d parts" % (len(err), len(parts)))
    print("    MAE %.2f deg, median %.2f, within 10 deg %.0f%%"
          % (err.mean(), np.median(err), 100 * np.mean(err < 10)))
    print("    held-out-by-object on T-LESS was %.2f deg" % model["held_out_mae"])
    print("    admitted by the gate: %.0f%%" % (100 * np.mean(dist <= model["gate"])))
    return err


def main():
    import joblib
    model = joblib.load(os.path.join(ROOT, "data", "tilt_model.joblib"))
    gate_on_real_photos(model)
    accuracy_on_renders(model)


if __name__ == "__main__":
    main()
