"""A pretraining set from SketchGraphs, rendered exactly as the PrintCAD set is."""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sketchgraphs_bridge as B  # noqa: E402
from sketchgraphs.data import flat_array  # noqa: E402
import sketchgraphs.data as sgd  # noqa: E402

WANT = int(os.environ.get("P2F_SG_N", "60000"))
SPLIT = os.environ.get("P2F_SG_SPLIT", "validation")


def main():
    path = os.path.join(B.SP, "sgdata", "sg_t16_%s.npy" % SPLIT)
    seqs = flat_array.load_dictionary_flat(path)["sequences"]
    imgs, rows = [], []
    for i in range(len(seqs)):
        if len(imgs) >= WANT:
            break
        try:
            got = B.convert(sgd.sketch_from_sequence(seqs[i]))
        except Exception:
            continue
        if got is None:
            continue
        img, r = got
        imgs.append((img * 255).astype(np.uint8))
        rows.append(r)
    X = np.stack(imgs)
    Y = np.stack(rows)
    out = os.path.join(ROOT, "data", "sketchnet_pretrain.npz")
    np.savez_compressed(out, X=X, Y=Y)
    n = Y[:, :, 0].sum(axis=1)
    print("%d sketches from SketchGraphs %s" % (len(X), SPLIT))
    print("  primitives per sketch: median %.0f, max %.0f" % (np.median(n), n.max()))
    print("  wrote %s (%.0f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
