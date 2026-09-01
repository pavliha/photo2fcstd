"""Does keeping the spatial feature map fix the localisation, holding everything else fixed?"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402
from photo2fcstd import sketchnet as SN  # noqa: E402
from sketchnet_geometry import BOX, endpoint_error, region_iou, tracer_on  # noqa: E402
from sketchnet_pretrain import DEV, fit  # noqa: E402


def evaluate(model, Xs, Ys, idx, filled=None, n=200):
    model.eval()
    ious, errs, right = [], [], []
    for i in idx[:n]:
        with torch.no_grad():
            raw = model(Xs[i][None].to(DEV)).cpu().numpy()[0]
        raw[:, 0] = 1 / (1 + np.exp(-raw[:, 0]))
        truth_rows = Ys[i].numpy()
        pred = SN.decode(raw, BOX)
        truth = SN.decode(truth_rows, BOX)
        ious.append(region_iou(pred, truth))
        errs.append(endpoint_error(raw, truth_rows))
        right.append(len(pred) == int(truth_rows[:, 0].sum()))
    return np.mean(ious), np.nanmedian(errs), np.mean(right)


def main():
    pre = np.load(os.path.join(ROOT, "data", "sketchnet_pretrain.npz"))
    Xp = torch.from_numpy(pre["X"].astype(np.float32) / 255.0).unsqueeze(1)
    Yp = torch.from_numpy(pre["Y"]).float()
    d = np.load(os.path.join(ROOT, "data", "sketchnet_tilt0.npz"), allow_pickle=True)
    X, Y, parts = d["X"], d["Y"], d["parts"]
    Xt = torch.from_numpy(X).float().unsqueeze(1)
    Yt = torch.from_numpy(Y).float()
    uniq = sorted(set(parts.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    train_parts = set(uniq[:int(0.8 * len(uniq))])
    tr = np.array([i for i, p in enumerate(parts) if p in train_parts])
    te = np.array([i for i, p in enumerate(parts) if p not in train_parts])
    pt = np.arange(int(0.98 * len(Xp)))

    for spatial in (False, True):
        m = SN.build_model(spatial=spatial).to(DEV)
        fit(m, Xp, Yp, pt, 6, 2e-3, tag="pre-%s" % ("spatial" if spatial else "pooled"))
        fit(m, Xt, Yt, tr, 12, 5e-4, tag="tune")
        iou, err, right = evaluate(m, Xt, Yt, te)
        print("\n  %-22s region IoU %.3f   endpoint error %.3f   right count %.0f%%\n"
              % ("spatial slots" if spatial else "global average", iou, err, 100 * right))
        if spatial:
            torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()},
                       SN.MODEL_PATH + ".spatial")
    print("  the geometric tracer, for comparison: region IoU 0.755, right count 40%")


if __name__ == "__main__":
    main()
