"""Pretrain on SketchGraphs, fine-tune on PrintCAD, and re-run gate 1.

sketchnet reached 96% correct primitive count on the parts it trained on and 40% across a part
split - it learns the mapping and cannot generalise from 967 parts. SketchGraphs supplies 60000 real
CAD sketches whose entities are the same primitives, rendered the same way, so the question is
whether the shortage was the only thing wrong.
"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402
from photo2fcstd import sketchnet as SN  # noqa: E402
from sketchnet_train import losses, tracer_baseline  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def fit(model, X, Y, idx, epochs, lr, batch=64, tag="", val=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(idx)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss, _, _ = losses(model(X[b].to(DEV)), Y[b].to(DEV), DEV)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        line = "  %-10s epoch %2d  train %.4f" % (tag, epoch, tot / len(perm))
        if val is not None:
            model.eval()
            with torch.no_grad():
                v = np.mean([float(losses(model(X[val[i:i + 128]].to(DEV)),
                                          Y[val[i:i + 128]].to(DEV), DEV)[0].detach())
                             for i in range(0, len(val), 128)])
            line += "  val %.4f" % v
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(line)
    return model


def counts_right(model, X, Y, idx):
    model.eval()
    got = []
    for i in range(0, len(idx), 128):
        b = idx[i:i + 128]
        with torch.no_grad():
            raw = model(X[b].to(DEV)).cpu().numpy()
        n = (1 / (1 + np.exp(-raw[..., 0])) > 0.5).sum(1)
        got.append(n == Y[b].numpy()[:, :, 0].sum(1))
    return float(np.concatenate(got).mean())


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
    n = len(Xp)
    pv = np.arange(int(0.98 * n), n)
    pt = np.arange(int(0.98 * n))
    print("pretrain on %d SketchGraphs sketches, fine-tune on %d PrintCAD renders (%d parts)\n"
          % (len(pt), len(tr), len(train_parts)))

    scratch = SN.build_model().to(DEV)
    fit(scratch, Xt, Yt, tr, 30, 2e-3, tag="scratch")
    a_tr, a_te = counts_right(scratch, Xt, Yt, tr), counts_right(scratch, Xt, Yt, te)

    model = SN.build_model().to(DEV)
    fit(model, Xp, Yp, pt, 8, 2e-3, tag="pretrain", val=pv)
    p_sg = counts_right(model, Xp, Yp, pv)
    p_te = counts_right(model, Xt, Yt, te)
    print("\n  after pretraining only: SketchGraphs held out %.0f%%, PrintCAD test %.0f%%"
          % (100 * p_sg, 100 * p_te))
    fit(model, Xt, Yt, tr, 15, 5e-4, tag="finetune")
    b_tr, b_te = counts_right(model, Xt, Yt, tr), counts_right(model, Xt, Yt, te)

    base = tracer_baseline(X[te][:200], Y[te][:200])
    tr_right = float(np.mean(base[:, 0] == base[:, 1]))
    print("\n  %-34s %8s %8s" % ("right number of primitives", "train", "test"))
    print("  %-34s %8s %7.0f%%" % ("geometric tracer", "-", 100 * tr_right))
    print("  %-34s %7.0f%% %7.0f%%" % ("sketchnet, from scratch", 100 * a_tr, 100 * a_te))
    print("  %-34s %7.0f%% %7.0f%%" % ("sketchnet, pretrained then tuned", 100 * b_tr, 100 * b_te))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               SN.MODEL_PATH + ".pretrained")


if __name__ == "__main__":
    main()
