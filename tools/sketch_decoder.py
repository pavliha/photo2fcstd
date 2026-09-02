"""Predict the ideal sketch as a raster from the photographs, then let the tracer vectorise it.

The tracer can only draw what survives segmentation, which is why most bad sketches have no
good trace from any photograph. Frozen features predict a part's primitive structure far
better than the tracer draws it, so this asks whether they can also place it: a small decoder
over DINOv3 patch tokens, trained against the sketch raster, which needs no correspondence
between predicted and true primitives.
"""
import json
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "src")
from photo2fcstd import patches, stats
from photo2fcstd.bench import photos_of

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Decoder(nn.Module):
    def __init__(self, channels=1024, views=3, width=192):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels * views, width, 1), nn.GELU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        blocks = []
        for _ in range(3):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        out = self.head(self.up(self.reduce(x)))
        return nn.functional.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)[:, 0]


def features(part):
    paths = sorted(photos_of(part))[:3]
    if len(paths) < 3:
        return None
    grids = patches.tokens_for(paths, allow_backbone=False)
    if grids is None or any(g is None for g in grids):
        return None
    return np.concatenate([g.transpose(2, 0, 1) for g in grids], axis=0)


def dataset(parts, rasters):
    X, Y, keep = [], [], []
    for part in parts:
        got = features(part)
        if got is None:
            continue
        X.append(got)
        Y.append(rasters[part])
        keep.append(part)
    return np.stack(X), np.stack(Y), keep


def iou(pred, truth, threshold=0.5):
    a, b = pred > threshold, truth > 0.5
    union = np.logical_or(a, b).sum(axis=(1, 2))
    inter = np.logical_and(a, b).sum(axis=(1, 2))
    return inter / np.maximum(union, 1)


def main():
    blob = np.load("data/sketch_rasters.npz", allow_pickle=True)
    rasters = {p: r.astype(np.float32) for p, r in zip(blob["parts"], blob["rasters"])}
    tune = [p for p in open("data/tune_subset.txt").read().split() if p in rasters]
    test = [p for p in open("data/test_ids.txt").read().split() if p in rasters]
    Xtr, Ytr, tune = dataset(tune, rasters)
    Xte, Yte, test = dataset(test, rasters)
    print("train %d, test %d, features %s" % (len(tune), len(test), Xtr.shape[1:]))
    model = Decoder(channels=Xtr.shape[1] // 3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, device=DEVICE)
    yt = torch.tensor(Ytr, device=DEVICE)
    for epoch in range(60):
        model.train()
        order = torch.randperm(len(xt), device=DEVICE)
        total = 0.0
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if (epoch + 1) % 15 == 0:
            print("  epoch %2d  loss %.4f" % (epoch + 1, total / len(xt)), flush=True)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(Xte, device=DEVICE))).cpu().numpy()
    mean_sketch = np.tile(Ytr.mean(axis=0), (len(Yte), 1, 1))
    got, base = iou(pred, Yte), iou(mean_sketch, Yte)
    print("\n  predicting the average sketch : IoU %.3f" % base.mean())
    print("  decoder from patch tokens     : IoU %.3f" % got.mean())
    r = stats.paired_delta({i: float(v) for i, v in enumerate(base)},
                           {i: float(v) for i, v in enumerate(got)})
    print("  %+.3f [%+.3f, %+.3f]%s" % (r["delta"], r["lo"], r["hi"],
          "" if r["significant"] else "  - indistinguishable from zero"))
    np.savez_compressed("data/sketch_decoder_preds.npz", parts=np.array(test), pred=pred.astype(np.float16))


if __name__ == "__main__":
    main()
