import os

import numpy as np

_MODEL = None


def available():
    from photo2fcstd.settings import PACKAGE_ROOT
    return os.path.exists(os.path.join(PACKAGE_ROOT, "data", "seq_model.pt"))


def model():
    global _MODEL
    if _MODEL is None:
        import torch
        from photo2fcstd import seqnet
        from photo2fcstd.settings import PACKAGE_ROOT
        m = seqnet.SeqNet().to(seqnet.device())
        m.load_state_dict(torch.load(os.path.join(PACKAGE_ROOT, "data", "seq_model.pt"),
                                     map_location=seqnet.device()))
        m.eval()
        _MODEL = m
    return _MODEL


def resample_map(contour, n):
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, np.concatenate([contour[:, 0], contour[:1, 0]]))
    y = np.interp(t, s, np.concatenate([contour[:, 1], contour[:1, 1]]))
    back = np.clip(np.searchsorted(s, t), 0, len(contour) - 1)
    return np.column_stack([x, y]), back


def spans_of_contour(contour):
    """Breakpoints and span types for one raw contour, as indices into that contour."""
    import torch
    from photo2fcstd import seqnet
    pts, back = resample_map(np.asarray(contour, float), seqnet.N_POINTS)
    centred = pts - pts.mean(axis=0)
    scale = float(np.abs(centred).max()) or 1.0
    x = seqnet.features((centred / scale)[None].astype(np.float32))
    spans = model().decode(torch.from_numpy(x).to(seqnet.device()))[0]
    out = sorted({(int(back[e]), t) for e, t in spans})
    dedup = []
    for e, t in out:
        if not dedup or e - dedup[-1][0] >= 2:
            dedup.append((e, t))
    return dedup


def seq_elements(raw, length_px):
    """The model's segmentation, fitted geometrically - a drop-in for trace.elements()."""
    from photo2fcstd.trace import arc_from_run, fit_circle, merge_and_snap, support_of
    raw = np.asarray(raw, float)
    spans = spans_of_contour(raw)
    if len(spans) >= 2:
        merged = []
        for e, typ in spans:
            if merged and typ == 1 and merged[-1][1] == 1:
                merged[-1] = (e, 1)
            else:
                merged.append((e, typ))
        if len(merged) >= 2 and merged[0][1] == 1 and merged[-1][1] == 1:
            merged[-1] = (merged[0][0], 1)
            merged.pop(0)
        spans = sorted(merged)
    if len(spans) == 1 and spans[0][1] == 1:
        return "round"
    if len(spans) < 2:
        return None
    els = []
    bps = [e for e, _ in spans]
    for i, (end, typ) in enumerate(spans):
        start = bps[i - 1]
        run = raw[start:end + 1] if end > start else np.vstack([raw[start:], raw[:end + 1]])
        if len(run) < 2:
            continue
        if typ == 1 and len(run) >= 5:
            cx, cy, r, rel, _ = fit_circle(run)
            if rel * r < max(0.05 * r, 3.0):
                els.append(arc_from_run(run))
                continue
        line = {"type": "line", "p0": run[0].tolist(), "p1": run[-1].tolist(), "_run": run}
        line["support"] = support_of(run, line, length_px)
        els.append(line)
    if len(els) < 2:
        return None
    for e in els:
        if "support" not in e and e.get("_run") is not None:
            e["support"] = support_of(e["_run"], e, length_px)
    return merge_and_snap(els, length_px)


def drawn_residual(els, raw, samples=6):
    pts = []
    for e in els:
        if e["type"] == "line":
            a, b = np.asarray(e["p0"], float), np.asarray(e["p1"], float)
            k = max(samples, int(np.hypot(*(b - a)) / 4) + 2)
            pts.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
        elif e["type"] == "arc":
            c = np.array([e["cx"], e["cy"]])
            k = max(6 * samples, int(2 * np.pi * e["r"] / 4) + 4)
            ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
            pts.append(c + e["r"] * np.column_stack([np.cos(ang), np.sin(ang)]))
    if not pts:
        return float("inf")
    drawn = np.vstack(pts)
    step = max(len(raw) // 200, 1)
    q = np.asarray(raw, float)[::step]
    d = np.sqrt(((q[:, None] - drawn[None]) ** 2).sum(-1)).min(1)
    return float(d.mean())
