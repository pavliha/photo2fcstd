"""Predict a sketch's primitives straight from a raster, with no correspondence to establish.

Ten learned attempts here have lost and nine needed the same thing: a mapping between what the
tracer produced and what the part actually is. Constraint prediction died at 62% alignment. Corner
heatmaps and per-point curve labels died the same way. Going raster to primitives sidesteps it - the
STEP files hold exact sequences and `synth` already renders them, so the supervision is free and
exact and no correspondence is needed at any point.

A fixed set of slots rather than a sequence, because PICASSO found a feed-forward set predictor beat
autoregressive Vitruvion 0.751 to 0.537, and because order is something a set predictor never has to
learn. Slots are matched to the truth by Hungarian assignment on a geometric cost.
"""
import os

import numpy as np

SLOTS = 24
TYPES = ("line", "arc", "circle")
SIDE = 128
PARAMS = 5
MODEL_PATH = os.environ.get("P2F_SKETCHNET", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "sketchnet.pt"))


def encode(els, box):
    """Primitives as rows of (present, type one-hot, five parameters) in a unit box.

    A line is its two endpoints; a circle is its centre and radius; an arc is its endpoints and the
    signed sagitta, which is finite for a straight arc and carries the bulge direction, unlike a
    centre that runs to infinity as an arc flattens.
    """
    lo, size = box
    unit = lambda p: (np.asarray(p, float) - lo) / max(size, 1e-9)
    rows = np.zeros((SLOTS, 1 + len(TYPES) + PARAMS), np.float32)
    for i, e in enumerate(els[:SLOTS]):
        kind = e.get("type", "line")
        rows[i, 0] = 1.0
        if kind == "circle":
            c = unit([e["cx"], e["cy"]])
            rows[i, 1 + TYPES.index("circle")] = 1.0
            rows[i, 4:9] = [c[0], c[1], float(e["r"]) / max(size, 1e-9), 0.0, 0.0]
            continue
        p0, p1 = unit(e["p0"]), unit(e["p1"])
        if kind == "arc" and e.get("r"):
            chord = float(np.linalg.norm(p1 - p0))
            r = float(e["r"]) / max(size, 1e-9)
            sag = r - np.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
            sag = sag if e.get("ccw", True) else -sag
            rows[i, 1 + TYPES.index("arc")] = 1.0
        else:
            sag = 0.0
            rows[i, 1 + TYPES.index("line")] = 1.0
        rows[i, 4:9] = [p0[0], p0[1], p1[0], p1[1], sag]
    return rows


def decode(rows, box, present=0.5):
    """Rows back to primitives in image coordinates."""
    lo, size = box
    out = []
    for r in np.asarray(rows, float):
        if r[0] < present:
            continue
        kind = TYPES[int(np.argmax(r[1:1 + len(TYPES)]))]
        p = r[4:9]
        if kind == "circle":
            out.append({"type": "circle", "cx": float(p[0] * size + lo[0]),
                        "cy": float(p[1] * size + lo[1]), "r": float(abs(p[2]) * size)})
            continue
        a = np.array([p[0], p[1]], float) * size + lo
        b = np.array([p[2], p[3]], float) * size + lo
        el = {"type": "line", "p0": a.tolist(), "p1": b.tolist()}
        if kind == "arc" and abs(p[4]) > 1e-4:
            chord = float(np.linalg.norm(b - a))
            sag = abs(float(p[4])) * size
            if chord > 1e-6 and sag > 1e-6:
                r = (sag * sag + (chord / 2) ** 2) / (2 * sag)
                mid = (a + b) / 2
                d = (b - a) / chord
                n = np.array([-d[1], d[0]])
                sign = 1.0 if p[4] > 0 else -1.0
                centre = mid + sign * n * (r - sag)
                el = {"type": "arc", "p0": a.tolist(), "p1": b.tolist(),
                      "cx": float(centre[0]), "cy": float(centre[1]), "r": float(r),
                      "ccw": bool(p[4] > 0)}
        out.append(el)
    return out


def geometric_cost(a, b):
    """How far one predicted row is from one true row, for the assignment."""
    same = 1.0 - float(np.dot(a[1:1 + len(TYPES)], b[1:1 + len(TYPES)]))
    ends = float(np.linalg.norm(a[4:8] - b[4:8]))
    flipped = float(np.linalg.norm(a[4:8] - np.concatenate([b[6:8], b[4:6]])))
    return min(ends, flipped) + 0.5 * same + abs(float(a[8] - b[8]))


def match(pred, true):
    """Hungarian assignment, so a set is not punished for arriving in a different order."""
    from scipy.optimize import linear_sum_assignment
    n = len(pred)
    cost = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            cost[i, j] = geometric_cost(pred[i], true[j]) if true[j, 0] > 0.5 else 0.0
    return linear_sum_assignment(cost)


def build_model(width=32, spatial=True):
    """A slot decoder attending over the feature map, not a global average.

    The first version ended its encoder with `AdaptiveAvgPool2d(1)` and regressed 24 primitives from
    the resulting single vector. It learned to emit the right *number* of primitives - 41% against
    the tracer's 40% - and put them in the wrong places, 0.32 of the sketch box out, for 0.455 region
    IoU against 0.755. Count survives global averaging and position does not, which is exactly the
    failure that shape predicts, and the reason DETR decodes slots by attending over the spatial map
    instead.
    """
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            def block(i, o, s=2):
                return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1),
                                     nn.GroupNorm(8, o), nn.GELU())
            w = width
            self.spatial = spatial
            self.body = nn.Sequential(block(1, w), block(w, w * 2), block(w * 2, w * 4),
                                      block(w * 4, w * 8))
            d = w * 8
            self.d = d
            if not spatial:
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1),
                                          nn.Linear(512, SLOTS * (1 + len(TYPES) + PARAMS)))
                return
            self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
            self.slots = nn.Parameter(torch.randn(1, SLOTS, d) * 0.02)
            layer = nn.TransformerDecoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d,
                                               dropout=0.1, batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(layer, num_layers=3)
            self.head = nn.Linear(d, 1 + len(TYPES) + PARAMS)

        def forward(self, x):
            f = self.body(x)
            if not self.spatial:
                y = self.head(self.pool(f).flatten(1))
                return y.reshape(-1, SLOTS, 1 + len(TYPES) + PARAMS)
            b = f.shape[0]
            tokens = f.flatten(2).transpose(1, 2) + self.pos[:, :f.shape[-1] * f.shape[-2]]
            q = self.slots.expand(b, -1, -1)
            return self.head(self.decoder(q, tokens))

    return Net()
