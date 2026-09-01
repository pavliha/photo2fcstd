"""Gate 1, judged on geometry rather than on how many primitives were emitted.

The first pass gated on primitive count, which a model can satisfy by emitting the right number of
arbitrary primitives - and that is exactly what happened. This scores the drawing: region IoU of the
predicted sketch against the real one, the same measure the rest of the project uses, plus the
median endpoint error of matched primitives.
"""
import os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402
from photo2fcstd import sketch_score as SS, sketchnet as SN  # noqa: E402

BOX = (np.zeros(2), 768.0)


def rings_of(els, step=12):
    """Primitives as polylines, so the drawing can be scored as a region."""
    out = []
    for e in els:
        if e["type"] == "circle":
            t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            out.append(np.column_stack([e["cx"] + e["r"] * np.cos(t), e["cy"] + e["r"] * np.sin(t)]))
        elif e["type"] == "arc":
            a0 = np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"])
            a1 = np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"])
            if e.get("ccw", True) and a1 < a0:
                a1 += 2 * np.pi
            if not e.get("ccw", True) and a1 > a0:
                a1 -= 2 * np.pi
            t = np.linspace(a0, a1, max(int(abs(a1 - a0) * 12), 3))
            out.append(np.column_stack([e["cx"] + e["r"] * np.cos(t), e["cy"] + e["r"] * np.sin(t)]))
        else:
            out.append(np.array([e["p0"], e["p1"]], float))
    return out


def region_iou(a, b):
    """Overlap of the two drawings, as closed regions, in the shared normalised frame."""
    pa, pb = SS.polygon_of([np.vstack(rings_of(a))]) if a else None, \
             SS.polygon_of([np.vstack(rings_of(b))]) if b else None
    if pa is None or pb is None or pa.is_empty or pb.is_empty:
        return 0.0
    return SS.region_iou(pa, pb)


def endpoint_error(pred_rows, true_rows):
    rows, cols = SN.match(pred_rows, true_rows)
    errs = [np.linalg.norm(pred_rows[i, 4:8] - true_rows[j, 4:8])
            for i, j in zip(rows, cols) if true_rows[j, 0] > 0.5 and pred_rows[i, 0] > 0.5]
    return float(np.median(errs)) if errs else np.nan


def tracer_on(filled):
    import cv2
    from photo2fcstd.trace import outline, primitives
    big = cv2.resize(filled, (768, 768), interpolation=cv2.INTER_NEAREST) > 0.5
    _, shape = outline(big)
    els = []
    for l in primitives([shape["raw"]] + shape["raw_holes"], 768.0):
        if l["type"] == "circle":
            els.append({"type": "circle", "cx": l["cx"], "cy": l["cy"], "r": l["r"]})
        else:
            els += l["elements"]
    return els


def main(n=250):
    stroke = np.load(os.path.join(ROOT, "data", "sketchnet_tilt0.npz"), allow_pickle=True)
    filled = np.load(os.path.join(ROOT, "data", "sketchnet_filled.npz"), allow_pickle=True)
    parts = stroke["parts"]
    uniq = sorted(set(parts.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[int(0.8 * len(uniq)):])
    te = np.array([i for i, p in enumerate(parts) if p in held])[:n]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SN.build_model().to(dev)
    m.load_state_dict(torch.load(SN.MODEL_PATH + ".pretrained", map_location=dev))
    m.eval()
    net_iou, tr_iou, net_err, net_cnt, tr_cnt = [], [], [], [], []
    for i in te:
        truth = SN.decode(stroke["Y"][i], BOX)
        with torch.no_grad():
            raw = m(torch.from_numpy(stroke["X"][i]).float()[None, None].to(dev)).cpu().numpy()[0]
        raw[:, 0] = 1 / (1 + np.exp(-raw[:, 0]))
        pred = SN.decode(raw, BOX)
        tr = tracer_on(filled["X"][i])
        net_iou.append(region_iou(pred, truth))
        tr_iou.append(region_iou(tr, truth))
        net_err.append(endpoint_error(raw, stroke["Y"][i]))
        nt = int(stroke["Y"][i][:, 0].sum())
        net_cnt.append(len(pred) == nt)
        tr_cnt.append(len(tr) == nt)
    print("gate 1 on geometry, n=%d held-out parts\n" % len(te))
    print("  %-24s %10s %10s %14s" % ("", "region IoU", "right count", "endpoint error"))
    print("  %-24s %10.3f %9.0f%% %14s" % ("geometric tracer", np.mean(tr_iou), 100 * np.mean(tr_cnt), "-"))
    print("  %-24s %10.3f %9.0f%% %14.3f"
          % ("sketchnet pretrained", np.mean(net_iou), 100 * np.mean(net_cnt), np.nanmedian(net_err)))
    print("\n  endpoint error is in units of the sketch box, so 0.28 means a typical predicted")
    print("  endpoint sits 28%% of the drawing's width from where it belongs")


if __name__ == "__main__":
    main()
