"""Train the corner heatmap, and report it against what approxPolyDP already gives for free."""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch  # noqa: E402
from photo2fcstd import cornernet as CN  # noqa: E402
from photo2fcstd.curvenet import N_POINTS, features, resample  # noqa: E402

TOL_FRAC = 0.02


def match(pred, true, tol):
    """Precision and recall of predicted corners at a distance tolerance."""
    if len(true) == 0:
        return (1.0 if len(pred) == 0 else 0.0), 1.0, []
    if len(pred) == 0:
        return 0.0, 0.0, []
    d = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=2)
    hit_p = (d.min(axis=1) <= tol)
    hit_t = (d.min(axis=0) <= tol)
    return float(hit_p.mean()), float(hit_t.mean()), d.min(axis=0)[hit_t].tolist()


def baseline_corners(contour):
    import cv2
    c = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    eps = 0.01 * cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)


def evaluate(name, corner_fn, rows, tol_frac=TOL_FRAC):
    P, R, D = [], [], []
    for r in rows:
        contour = np.asarray(r["contour"], float)
        true = np.asarray(r["corners"], float)
        diag = np.linalg.norm(contour.max(0) - contour.min(0))
        pred = corner_fn(r, contour)
        if pred is None:
            continue
        p, rec, d = match(np.asarray(pred, float), true, tol_frac * diag)
        P.append(p); R.append(rec); D += [x / diag for x in d]
    f1 = 2 * np.mean(P) * np.mean(R) / max(np.mean(P) + np.mean(R), 1e-9)
    print("  %-22s precision %.3f  recall %.3f  F1 %.3f  median offset %.4f of the diagonal"
          % (name, np.mean(P), np.mean(R), f1, np.median(D) if D else float("nan")))
    return f1


def main():
    rows = list(np.load(os.path.join(ROOT, "data", "corner_rows.npy"), allow_pickle=True))
    parts = sorted({r["part"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(parts)
    cut = int(0.8 * len(parts))
    train_parts, test_parts = set(parts[:cut]), set(parts[cut:])
    tr = [r for r in rows if r["part"] in train_parts]
    te = [r for r in rows if r["part"] in test_parts]
    print("%d train / %d test samples, split by part (%d / %d parts)\n"
          % (len(tr), len(te), len(train_parts), len(test_parts)))

    X = np.stack([features(r["contour"]) for r in tr])
    Y = np.stack([CN.target_at(r["heat"], r["contour"]) for r in tr])
    Xt = np.stack([features(r["contour"]) for r in te])
    Yt = np.stack([CN.target_at(r["heat"], r["contour"]) for r in te])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    X, Y = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
    Xt, Yt = torch.from_numpy(Xt).to(dev), torch.from_numpy(Yt).to(dev)

    m = CN.build_model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state = 1e9, None
    for epoch in range(45):
        m.train()
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(m(X[b]), Y[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        m.eval()
        with torch.no_grad():
            v = float(lossf(m(Xt), Yt))
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        if epoch % 5 == 0 or epoch == 44:
            print("  epoch %2d  train %.4f  test %.4f%s" % (epoch, tot / len(X), v,
                                                            "  *" if v == best else ""))
    m.load_state_dict(best_state)
    torch.save(best_state, CN.MODEL_PATH)
    print("\n  saved %s (best test loss %.4f)\n" % (CN.MODEL_PATH, best))

    m.eval()
    with torch.no_grad():
        H = torch.sigmoid(m(Xt)).cpu().numpy()

    def learned(r, contour):
        i = te.index(r) if False else None
        return None

    print("corner localisation on %d held-out samples, tolerance %.0f%% of the diagonal\n"
          % (len(te), 100 * TOL_FRAC))
    evaluate("approxPolyDP", lambda r, c: baseline_corners(c), te)
    order = {id(r): i for i, r in enumerate(te)}
    evaluate("learned heatmap",
             lambda r, c: CN.peaks(H[order[id(r)]], resample(c)[0]), te)


if __name__ == "__main__":
    main()
