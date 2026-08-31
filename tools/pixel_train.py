"""Does a photo say which view is squarest, when a silhouette cannot?

Silhouette statistics agree with the best view 47% of the time against 33% for chance, and that
was worth +0.038 end to end. Segmentation throws away shading across a face, specular highlights
and the ellipticity of an oblique hole, which is where foreshortening actually lives. Same target,
same split-by-part protocol, pixels instead of the mask.
"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DATA = os.environ.get("P2F_PIXEL_DATA", os.path.join(ROOT, "data", "pixel_views.npz"))
OUT = os.path.join(ROOT, "data", "pixel_view.pt")


def build_model(w=24):
    def block(i, o, s=2):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1),
                             nn.GroupNorm(4, o), nn.GELU())
    return nn.Sequential(block(2, w), block(w, w * 2), block(w * 2, w * 4),
                         block(w * 4, w * 4), block(w * 4, w * 4),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(w * 4, 1))


def augment(x, rng):
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(rng.integers(0, 4))
    return torch.rot90(x, k, dims=[-2, -1])


def main():
    d = np.load(DATA, allow_pickle=True)
    X, iou, parts = d["X"], d["iou"], d["parts"]
    n = len(X)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    a, b = int(0.65 * n), int(0.80 * n)
    tr, va, te = order[:a], order[a:b], order[b:]
    print("%d parts, %d train / %d val / %d test, split by part" % (n, len(tr), len(va), len(te)))
    print("the epoch is chosen on val and reported on test, which the first run got wrong\n")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.from_numpy(X).float()
    y = torch.from_numpy((iou == iou.max(axis=1, keepdims=True)).astype(np.float32))

    m = build_model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = -1, None
    g = np.random.default_rng(1)
    for epoch in range(60):
        m.train()
        for i in range(0, len(tr), 16):
            b = tr[i:i + 16]
            xb = torch.stack([augment(Xt[j], g) for j in b]).to(dev)
            yb = y[b].to(dev)
            opt.zero_grad()
            loss = lossf(m(xb.reshape(-1, 2, 128, 128)).reshape(len(b), 3), yb)
            loss.backward()
            opt.step()
        m.eval()
        def run(idx):
            with torch.no_grad():
                out = m(Xt[idx].reshape(-1, 2, 128, 128).to(dev)).reshape(len(idx), 3).cpu().numpy()
            pick = out.argmax(1)
            return (float(np.mean(pick == iou[idx].argmax(1))),
                    float(np.mean(iou[idx][np.arange(len(idx)), pick])))
        vacc, vgot = run(va)
        if vgot > best:
            best = vgot
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if epoch % 10 == 0 or epoch == 59:
            print("  epoch %2d  val agree %.2f  val IoU %.3f" % (epoch, vacc, vgot))

    m.load_state_dict(best_state)
    m.eval()
    torch.save(best_state, OUT)
    with torch.no_grad():
        out = m(Xt[te].reshape(-1, 2, 128, 128).to(dev)).reshape(len(te), 3).cpu().numpy()
    pick = out.argmax(1)
    print("\n  held-out test, n=%d parts\n" % len(te))
    print("  %-26s %8s %8s" % ("chooser", "agree", "IoU"))
    print("  %-26s %8.2f %8.3f" % ("chance", 1 / 3, float(np.mean(iou[te]))))
    print("  %-26s %8s %8.3f" % ("first photo", "-", float(np.mean(iou[te][:, 0]))))
    print("  %-26s %8.2f %8.3f" % ("pixels", float(np.mean(pick == iou[te].argmax(1))),
                                   float(np.mean(iou[te][np.arange(len(te)), pick]))))
    print("  %-26s %8.2f %8.3f" % ("oracle", 1.0, float(np.mean(iou[te].max(1)))))
    print("\n  saved", OUT)


if __name__ == "__main__":
    main()
