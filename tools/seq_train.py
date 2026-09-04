"""Train the breakpoint model. Split by design/part, checkpoint on best held-out token accuracy."""
import os, sys, time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import seqnet  # noqa: E402


def load(path):
    d = np.load(path, allow_pickle=True)
    X = seqnet.features(d["X"])
    seqs = d["seq"]
    groups = d["group"]
    T = np.full((len(seqs), seqnet.MAX_LEN), -100, np.int64)
    for i, s in enumerate(seqs):
        toks = [seqnet.TOK_LINE] + seqnet.encode_target(s)
        T[i, :len(toks)] = toks[:seqnet.MAX_LEN]
    return X, T, groups


def split(groups, frac=0.1, seed=0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    val = set(uniq[rng.choice(len(uniq), int(frac * len(uniq)), replace=False)])
    v = np.array([g in val for g in groups])
    return ~v, v


def token_acc(model, X, T, dev, batch=256):
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(dev)
            t = torch.from_numpy(T[i:i + batch]).to(dev)
            logits = model(x, t[:, :-1])
            tgt = t[:, 1:]
            keep = tgt != -100
            ok += int((logits.argmax(-1)[keep] == tgt[keep]).sum())
            n += int(keep.sum())
    model.train()
    return ok / max(n, 1)


def main(path="data/seq_data.npz", epochs=30, batch=96, lr=3e-4):
    dev = seqnet.device()
    X, T, groups = load(os.path.join(ROOT, path))
    tr, va = split(groups)
    print("train %d, val %d (by %d groups), device %s" % (tr.sum(), va.sum(), len(np.unique(groups)), dev))
    model = seqnet.SeqNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    Xt, Tt = X[tr], T[tr]
    best = 0.0
    for ep in range(epochs):
        order = np.random.permutation(len(Xt))
        tl = 0.0
        t0 = time.time()
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.from_numpy(Xt[idx]).to(dev)
            t = torch.from_numpy(Tt[idx]).to(dev)
            logits = model(x, t[:, :-1])
            loss = loss_fn(logits.reshape(-1, seqnet.VOCAB), t[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss) * len(idx)
        sched.step()
        acc = token_acc(model, X[va], T[va], dev)
        mark = ""
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(ROOT, "data", "seq_model.pt"))
            mark = "  <- saved"
        print("epoch %2d  loss %.4f  val token acc %.4f  (%.0fs)%s" % (ep, tl / len(Xt), acc, time.time() - t0, mark))
    print("best val token accuracy %.4f" % best)


if __name__ == "__main__":
    main(*sys.argv[1:2], **{k: (float(v) if "." in v else int(v)) for k, v in
                            (a.split("=") for a in sys.argv[2:])})
