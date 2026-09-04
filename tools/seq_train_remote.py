import os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace")
import seqnet


def load(path):
    d = np.load(path, allow_pickle=True)
    X = seqnet.features(d["X"])
    T = np.full((len(d["seq"]), seqnet.MAX_LEN), -100, np.int64)
    for i, s in enumerate(d["seq"]):
        toks = [seqnet.TOK_LINE] + seqnet.encode_target(s)
        T[i, :len(toks)] = toks[:seqnet.MAX_LEN]
    return X, T, d["group"]


def split(groups, frac=0.1, seed=0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    val = set(uniq[rng.choice(len(uniq), int(frac * len(uniq)), replace=False)])
    v = np.array([g in val for g in groups])
    return ~v, v


def token_acc(model, X, T, dev, batch=512):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(dev)
            t = torch.from_numpy(T[i:i + batch]).to(dev)
            logits = model(x, t[:, :-1])
            keep = t[:, 1:] != -100
            ok += int((logits.argmax(-1)[keep] == t[:, 1:][keep]).sum()); n += int(keep.sum())
    model.train()
    return ok / max(n, 1)


def train(Xt, Tt, Xv, Tv, Xrv, Trv, tag, epochs=80, batch=256, lr=3e-4):
    dev = "cuda"
    model = seqnet.SeqNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best = 0.0
    for ep in range(epochs):
        order = np.random.permutation(len(Xt))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.from_numpy(Xt[idx]).to(dev)
            t = torch.from_numpy(Tt[idx]).to(dev)
            loss = loss_fn(model(x, t[:, :-1]).reshape(-1, seqnet.VOCAB), t[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = token_acc(model, Xrv, Trv, dev)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "/workspace/out/%s.pt" % tag)
        if ep % 10 == 9 or ep == epochs - 1:
            print("  %s ep %2d  real-val %.4f  synth-val %.4f" % (tag, ep, acc, token_acc(model, Xv, Tv, dev)), flush=True)
    print("%s best real-val token acc %.4f" % (tag, best), flush=True)
    return best


def main():
    os.makedirs("/workspace/out", exist_ok=True)
    Xs, Ts, Gs = load("/workspace/seq_data4.npz")
    Xr, Tr, Gr = load("/workspace/seq_real.npz")
    str_, svl = split(Gs)
    rtr, rvl = split(Gr)
    print("synthetic %d train / %d val, real %d train / %d val" % (str_.sum(), svl.sum(), rtr.sum(), rvl.sum()), flush=True)
    L = max(Ts.shape[1], Tr.shape[1])
    pad = lambda T: np.pad(T, ((0, 0), (0, L - T.shape[1])), constant_values=-100)
    Ts, Tr = pad(Ts), pad(Tr)
    for mult in (0, 4, 12, 30):
        Xt = np.concatenate([Xs[str_]] + [Xr[rtr]] * mult) if mult else Xs[str_]
        Tt = np.concatenate([Ts[str_]] + [Tr[rtr]] * mult) if mult else Ts[str_]
        train(Xt, Tt, Xs[svl], Ts[svl], Xr[rvl], Tr[rvl], "mix%d" % mult)


if __name__ == "__main__":
    main()
