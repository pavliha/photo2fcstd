import os

import numpy as np

N_POINTS = 1024
SCALES = (1, 2, 4, 8, 16, 32, 64)
MODEL_PATH = os.environ.get("P2F_CURVENET", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "curvenet.pt"))


def resample(contour, n=N_POINTS):
    p = np.asarray(contour, float)
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0), np.zeros(n, int)
    t = np.linspace(0, s[-1], n, endpoint=False)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    frac = ((t - s[idx]) / np.maximum(seg[idx], 1e-9))[:, None]
    return closed[idx] + frac * (closed[idx + 1] - closed[idx]), idx


def features(contour, n=N_POINTS):
    p, _ = resample(contour, n)
    c = p - p.mean(axis=0)
    scale = max(np.abs(c).max(), 1e-9)
    c = c / scale
    ch = []
    for s in SCALES:
        a = np.roll(c, s, axis=0) - c
        b = np.roll(c, -s, axis=0) - c
        na = np.linalg.norm(a, axis=1) + 1e-9
        nb = np.linalg.norm(b, axis=1) + 1e-9
        cosang = np.sum(a * b, axis=1) / (na * nb)
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        sinang = cross / (na * nb)
        chord = np.linalg.norm(np.roll(c, -s, axis=0) - np.roll(c, s, axis=0), axis=1)
        arc = na + nb
        ch += [cosang, sinang, chord / np.maximum(arc, 1e-9)]
    return np.asarray(ch, np.float32)


def labels_at(label, contour, n=N_POINTS, amb=None):
    _, idx = resample(contour, n)
    i = np.clip(idx, 0, len(label) - 1)
    y = np.asarray(label, np.int64)[i]
    if amb is not None:
        y = np.where(np.asarray(amb, bool)[i], -100, y)
    return y


def build_model(nclass=2):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, cin=len(SCALES) * 3, h=160, nclass=2):
            super().__init__()
            def block(i, o, d):
                return nn.Sequential(nn.Conv1d(i, o, 5, padding=2 * d, dilation=d, padding_mode="circular"),
                                     nn.GroupNorm(8, o), nn.GELU())
            self.body = nn.Sequential(block(cin, h, 1), block(h, h, 2), block(h, h, 4),
                                      block(h, h, 8), block(h, h, 16), block(h, h, 32))
            self.head = nn.Conv1d(h, nclass, 1)

        def forward(self, x):
            return self.head(self.body(x))

    return Net(nclass=nclass)


def load(path=MODEL_PATH):
    import torch
    if not os.path.exists(path):
        return None, None
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = build_model()
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.to(dev).eval(), dev


_CACHE = {}


def predict(contour):
    import torch
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["d"] = load()
    m, dev = _CACHE["m"], _CACHE["d"]
    if m is None:
        return None
    x = torch.from_numpy(features(contour)).unsqueeze(0).to(dev)
    with torch.no_grad():
        y = m(x)[0].argmax(0).cpu().numpy()
    _, idx = resample(contour)
    out = np.zeros(len(contour), int)
    counts = np.zeros((len(contour), 3), int)
    for j, i in enumerate(idx):
        counts[i, y[j]] += 1
    seen = counts.sum(axis=1) > 0
    out[seen] = counts[seen].argmax(axis=1)
    if (~seen).any():
        fill = np.flatnonzero(seen)
        if len(fill):
            out[~seen] = out[fill[np.searchsorted(fill, np.flatnonzero(~seen)).clip(0, len(fill) - 1)]]
    return out
