"""Predict where the corners of a traced outline are, as peaks on the contour.

Labelling every contour point straight or curved reached 0.82 accuracy and lost end to end,
0.579 to 0.452 sketch IoU, because a per-point boundary is fuzzy by several points and a
corner off by a few points moves a line endpoint visibly. The output here is a heatmap whose
peaks decode to a position between two contour points, which is the same thing `approxPolyDP`
produces and the same thing the primitive fitter consumes.
"""
import os

import numpy as np

from photo2fcstd.curvenet import N_POINTS, SCALES, features, resample

MODEL_PATH = os.environ.get("P2F_CORNERNET", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cornernet.pt"))
PEAK_THRESHOLD = 0.35
NMS_POINTS = 12
_CACHE = {}


def target_at(heat, contour, n=N_POINTS):
    _, idx = resample(contour, n)
    return np.asarray(heat, np.float32)[np.clip(idx, 0, len(heat) - 1)]


def build_model(h=160):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, cin=len(SCALES) * 3):
            super().__init__()
            def block(i, o, d):
                return nn.Sequential(nn.Conv1d(i, o, 5, padding=2 * d, dilation=d,
                                               padding_mode="circular"),
                                     nn.GroupNorm(8, o), nn.GELU())
            self.body = nn.Sequential(block(cin, h, 1), block(h, h, 2), block(h, h, 4),
                                      block(h, h, 8), block(h, h, 16), block(h, h, 32))
            self.head = nn.Conv1d(h, 1, 1)

        def forward(self, x):
            return self.head(self.body(x)).squeeze(1)

    return Net()


def peaks(heat, points, threshold=PEAK_THRESHOLD, nms=NMS_POINTS):
    """Local maxima above a floor, each refined to a weighted centroid of its neighbourhood."""
    h = np.asarray(heat, float)
    n = len(h)
    if n == 0:
        return np.zeros((0, 2))
    keep = (h >= threshold) & (h >= np.roll(h, 1)) & (h >= np.roll(h, -1))
    order = np.argsort(-h)
    taken, out = np.zeros(n, bool), []
    for i in order:
        if not keep[i] or taken[i]:
            continue
        lo = (np.arange(-nms, nms + 1) + i) % n
        w = h[lo]
        w = np.where(w > 0, w, 0)
        if w.sum() <= 0:
            continue
        out.append((points[lo] * w[:, None]).sum(axis=0) / w.sum())
        taken[lo] = True
    return np.asarray(out) if out else np.zeros((0, 2))


def load(path=MODEL_PATH):
    import torch
    if not os.path.exists(path):
        return None, None
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = build_model()
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.to(dev).eval(), dev


def predict(contour):
    """Corner positions in contour coordinates, or None when no model is installed."""
    import torch
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["d"] = load()
    m, dev = _CACHE["m"], _CACHE["d"]
    if m is None:
        return None
    x = torch.from_numpy(features(contour)).unsqueeze(0).to(dev)
    with torch.no_grad():
        h = torch.sigmoid(m(x))[0].cpu().numpy()
    pts, _ = resample(contour)
    return peaks(h, pts)


def heatmap(contour):
    """The predicted corner score at each point of the original contour."""
    import torch
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["d"] = load()
    m, dev = _CACHE["m"], _CACHE["d"]
    if m is None:
        return None
    x = torch.from_numpy(features(contour)).unsqueeze(0).to(dev)
    with torch.no_grad():
        h = torch.sigmoid(m(x))[0].cpu().numpy()
    _, idx = resample(contour)
    out = np.zeros(len(contour))
    np.maximum.at(out, np.clip(idx, 0, len(contour) - 1), h)
    return out
