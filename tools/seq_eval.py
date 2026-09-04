"""The proxy eval, with the shipped tracer as the baseline on the same held-out samples.

Metrics: breakpoint F1 (a predicted breakpoint within +-3 resampled points of a true one) and
per-point span-type accuracy. The baseline runs `elements()` - corner_runs plus the arc gates,
chain_arcs included - on the same resampled contour and converts its runs to the same format.
"""
import os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import seqnet  # noqa: E402
from seq_train import load, split  # noqa: E402


def span_points(spans, n=seqnet.N_POINTS):
    y = np.zeros(n, int)
    prev = 0
    for e, t in spans:
        y[prev:min(e, n - 1) + 1] = t
        prev = min(e, n - 1) + 1
    return y


def spans_from_seq(s):
    out, j = [], 0
    while j + 1 < len(s) and s[j] >= 0:
        out.append((int(s[j + 1]), int(s[j])))
        j += 2
    return out


def bp_f1(pred, true, tol=3, n=seqnet.N_POINTS):
    p = sorted(e for e, _ in pred)
    t = sorted(e for e, _ in true)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    def circ(a, b):
        d = abs(a - b) % n
        return min(d, n - d)
    hit_p = sum(1 for a in p if any(circ(a, b) <= tol for b in t))
    hit_t = sum(1 for b in t if any(circ(a, b) <= tol for a in p))
    prec, rec = hit_p / len(p), hit_t / len(t)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def tracer_spans(x_norm):
    from photo2fcstd.trace import elements
    pts = x_norm * 300.0 + 380.0
    length = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    els = elements(pts, length)
    spans = []
    cursor = 0
    for e in els:
        run = e.get("_run")
        k = len(run) if run is not None else 2
        end = min(cursor + max(k - 1, 1), seqnet.N_POINTS - 1)
        spans.append((end, 1 if e["type"] == "arc" else 0))
        cursor = end
    if spans:
        spans[-1] = (seqnet.N_POINTS - 1, spans[-1][1])
    return spans


def tracer_spans_exact(x_norm):
    from photo2fcstd.trace import elements
    pts = x_norm * 300.0 + 380.0
    length = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    els = elements(pts, length)
    spans = []
    for e in els:
        tail = np.asarray(e["p1"], float)
        d = np.linalg.norm(pts - tail, axis=1)
        spans.append((int(d.argmin()), 1 if e["type"] == "arc" else 0))
    return sorted(set(spans), key=lambda s: s[0])


def main(path="data/seq_data.npz", limit=800):
    limit = int(limit)
    dev = seqnet.device()
    d = np.load(os.path.join(ROOT, path), allow_pickle=True)
    Xraw, seqs, groups = d["X"], d["seq"], d["group"]
    tr, va = split(groups)
    idx = np.where(va)[0][:limit]
    model = seqnet.SeqNet().to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "data", "seq_model.pt"), map_location=dev))
    model.eval()
    X = seqnet.features(Xraw[idx])
    rows = {"model": [], "tracer": []}
    B = 128
    preds = []
    for i in range(0, len(idx), B):
        preds += model.decode(torch.from_numpy(X[i:i + B]).to(dev))
    for k, i in enumerate(idx):
        true = spans_from_seq(seqs[i])
        ty = span_points(true)
        for name, spans in (("model", preds[k]), ("tracer", tracer_spans_exact(Xraw[i]))):
            rows[name].append((bp_f1(spans, true),
                               float((span_points(spans) == ty).mean()),
                               abs(len(spans) - len(true))))
    print("held-out samples: %d   (curved fraction of points: %.2f)"
          % (len(idx), np.mean([span_points(spans_from_seq(seqs[i])).mean() for i in idx])))
    print("\n  %-8s %14s %18s %16s" % ("", "breakpoint F1", "span-type acc", "|d span count|"))
    for name in ("tracer", "model"):
        r = np.array(rows[name])
        print("  %-8s %14.3f %18.3f %16.2f" % (name, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean()))


if __name__ == "__main__":
    main(*sys.argv[1:])
