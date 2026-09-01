"""Are the tracer and the model complementary? Measure the ceiling before building a chooser.

Both learned components that work in this project select among candidates; all eleven that failed
tried to replace a geometric step. The tracer fragments badly on some parts - 58 primitives against
a truth of 11 on 00042 - and the model degrades differently, so the useful question is whether
choosing between them per part beats either alone.
"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402
from photo2fcstd import sketchnet as SN, stats  # noqa: E402
from sketchnet_geometry import BOX, region_iou, tracer_on  # noqa: E402


def main(n=250):
    stroke = np.load(os.path.join(ROOT, "data", "sketchnet_tilt0.npz"), allow_pickle=True)
    filled = np.load(os.path.join(ROOT, "data", "sketchnet_filled.npz"), allow_pickle=True)
    parts = stroke["parts"]
    uniq = sorted(set(parts.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[int(0.8 * len(uniq)):])
    te = np.array([i for i, p in enumerate(parts) if p in held])[:n]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # the encoder changed when global pooling was replaced, so the older checkpoint no longer
    # matches; take whichever of the current architectures has weights on disk
    # the pooled encoder beat the spatial one on geometry, 0.409 against 0.358 region IoU
    path, spatial = SN.MODEL_PATH + ".pooled", False
    if not os.path.exists(path):
        raise SystemExit("no current checkpoint yet - run tools/sketchnet_spatial_ab.py first")
    m = SN.build_model(spatial=spatial).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    rows = []
    for i in te:
        truth = SN.decode(stroke["Y"][i], BOX)
        with torch.no_grad():
            raw = m(torch.from_numpy(stroke["X"][i]).float()[None, None].to(dev)).cpu().numpy()[0]
        raw[:, 0] = 1 / (1 + np.exp(-raw[:, 0]))
        tr = tracer_on(filled["X"][i])
        rows.append({"part": str(parts[i]),
                     "tracer": region_iou(tr, truth), "net": region_iou(SN.decode(raw, BOX), truth),
                     "tracer_n": len(tr), "truth_n": len(truth)})
    t = np.array([r["tracer"] for r in rows])
    nt = np.array([r["net"] for r in rows])
    best = np.maximum(t, nt)
    print("n=%d held-out parts\n" % len(rows))
    print("  %-28s %10s" % ("", "region IoU"))
    print("  %-28s %10.3f" % ("geometric tracer", t.mean()))
    print("  %-28s %10.3f" % ("sketchnet", nt.mean()))
    print("  %-28s %10.3f" % ("best of the two, per part", best.mean()))
    m_, lo, hi = stats.mean_ci(best - t)
    print("\n  headroom over the tracer: %+.4f [%+.4f, %+.4f]" % (m_, lo, hi))
    print("  the model wins on %.0f%% of parts" % (100 * np.mean(nt > t)))
    frag = np.array([r["tracer_n"] > 2 * max(r["truth_n"], 1) for r in rows])
    if frag.any():
        print("\n  where the tracer fragments (over twice the true count, %.0f%% of parts):" % (100 * frag.mean()))
        print("    tracer %.3f, sketchnet %.3f, model wins %.0f%%"
              % (t[frag].mean(), nt[frag].mean(), 100 * np.mean(nt[frag] > t[frag])))
        print("  where it does not:")
        print("    tracer %.3f, sketchnet %.3f, model wins %.0f%%"
              % (t[~frag].mean(), nt[~frag].mean(), 100 * np.mean(nt[~frag] > t[~frag])))


if __name__ == "__main__":
    main()
