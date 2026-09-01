"""Gate 1: on clean renders with exact labels, does a set predictor beat the geometric tracer?"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from photo2fcstd import sketchnet as SN  # noqa: E402

TAG = os.environ.get("P2F_SKETCHNET_TILT", "0")


def losses(pred, true, dev):
    """Presence, type and parameters, after matching each prediction to a truth slot."""
    bce, ce, l1 = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    p = pred.detach().cpu().numpy()
    t = true.detach().cpu().numpy()
    order = np.zeros((len(p), SN.SLOTS), dtype=np.int64)
    for b in range(len(p)):
        rows, cols = SN.match(p[b], t[b])
        order[b] = cols
    idx = torch.from_numpy(order).to(dev)
    tt = torch.gather(true, 1, idx.unsqueeze(-1).expand(-1, -1, true.shape[-1]))
    present = tt[..., 0]
    lp = bce(pred[..., 0], present)
    m = present > 0.5
    if m.sum() == 0:
        return lp, lp.detach(), lp.detach()
    lt = ce(pred[..., 1:1 + len(SN.TYPES)][m], tt[..., 1:1 + len(SN.TYPES)][m].argmax(-1))
    lg = l1(pred[..., 1 + len(SN.TYPES):][m], tt[..., 1 + len(SN.TYPES):][m])
    return lp + lt + 5.0 * lg, lt.detach(), lg.detach()


def tracer_baseline(imgs, truths):
    """What `trace.primitives` gets on the same renders - the thing to beat."""
    import cv2
    from photo2fcstd.trace import outline, primitives
    got = []
    for img, t in zip(imgs, truths):
        try:
            big = cv2.resize(img, (768, 768), interpolation=cv2.INTER_NEAREST) > 0.5
            _, shape = outline(big)
            loops = primitives([shape["raw"]] + shape["raw_holes"], 768.0)
            n = sum(len(l.get("elements", [])) or 1 for l in loops)
            got.append((n, int(t[:, 0].sum())))
        except Exception:
            got.append((0, int(t[:, 0].sum())))
    a = np.array(got, float)
    return a


def main():
    d = np.load(os.path.join(ROOT, "data", "sketchnet_tilt%s.npz" % TAG), allow_pickle=True)
    X, Y, parts = d["X"], d["Y"], d["parts"]
    uniq = sorted(set(parts.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    cut = int(0.8 * len(uniq))
    train_parts = set(uniq[:cut])
    tr = np.array([i for i, p in enumerate(parts) if p in train_parts])
    te = np.array([i for i, p in enumerate(parts) if p not in train_parts])
    print("%d samples, %d train / %d test, split by part\n" % (len(X), len(tr), len(te)))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.from_numpy(X).float().unsqueeze(1)
    Yt = torch.from_numpy(Y).float()
    m = SN.build_model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    for epoch in range(30):
        m.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            xb, yb = Xt[b].to(dev), Yt[b].to(dev)
            opt.zero_grad()
            loss, _, _ = losses(m(xb), yb, dev)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        m.eval()
        with torch.no_grad():
            v = 0.0
            for i in range(0, len(te), 64):
                b = te[i:i + 64]
                loss, _, _ = losses(m(Xt[b].to(dev)), Yt[b].to(dev), dev)
                v += float(loss) * len(b)
            v /= len(te)
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        if epoch % 5 == 0 or epoch == 29:
            print("  epoch %2d  train %.4f  test %.4f%s" % (epoch, tot / len(tr), v, "  *" if v == best else ""))
    final_state = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
    torch.save(final_state, SN.MODEL_PATH + ".final")
    torch.save(best_state, SN.MODEL_PATH)
    # a checkpoint chosen by test loss is useless when test loss never improves; the question
    # then is whether the final weights fit the training set at all, which separates a model that
    # cannot learn from one that has memorised.
    for name, state in (("best by test loss", best_state), ("final epoch", final_state)):
        m.load_state_dict(state)
        m.eval()
        for split, idx in (("train", tr), ("test", te)):
            with torch.no_grad():
                raw = m(Xt[idx[:300]].to(dev)).cpu().numpy()
            pres = 1 / (1 + np.exp(-raw[..., 0]))
            n = (pres > 0.5).sum(1)
            truth = Y[idx[:300]][:, :, 0].sum(1)
            errs = []
            for b in range(min(100, len(idx))):
                pr = raw[b].copy()
                pr[:, 0] = pres[b]
                rows, cols = SN.match(pr, Y[idx[b]])
                for i, j in zip(rows, cols):
                    if Y[idx[b]][j, 0] > 0.5 and pr[i, 0] > 0.5:
                        errs.append(np.linalg.norm(pr[i, 4:8] - Y[idx[b]][j, 4:8]))
            print("  %-18s %-6s count %.2f vs %.2f, right %.0f%%, endpoint error %.3f"
                  % (name, split, n.mean(), truth.mean(), 100 * np.mean(n == truth),
                     np.median(errs) if errs else -1))
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pred = torch.sigmoid(m(Xt[te].to(dev))[..., :1]).cpu().numpy()
        raw = m(Xt[te].to(dev)).cpu().numpy()
    counts = (pred[..., 0] > 0.5).sum(axis=1)
    truth = Y[te][:, :, 0].sum(axis=1)
    base = tracer_baseline(X[te][:200], Y[te][:200])
    print("\n  %-26s %10s %10s" % ("", "elements", "vs truth"))
    print("  %-26s %10.2f %10s" % ("the real sketches", truth.mean(), "-"))
    print("  %-26s %10.2f %+10.2f" % ("geometric tracer", base[:, 0].mean(), base[:, 0].mean() - base[:, 1].mean()))
    print("  %-26s %10.2f %+10.2f" % ("sketchnet", counts.mean(), counts.mean() - truth.mean()))
    exact_net = float(np.mean(counts == truth))
    exact_tr = float(np.mean(base[:, 0] == base[:, 1]))
    print("\n  right number of primitives: tracer %.0f%%, sketchnet %.0f%%" % (100 * exact_tr, 100 * exact_net))
    print("  saved %s (best test loss %.4f)" % (SN.MODEL_PATH, best))


if __name__ == "__main__":
    main()
