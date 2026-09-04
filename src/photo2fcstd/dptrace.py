import os

import numpy as np

DP_POINTS = int(os.environ.get("P2F_DP_POINTS", 256))
DP_SIGMA = float(os.environ.get("P2F_DP_SIGMA", 1.6))
DP_ELEMENT_COST = float(os.environ.get("P2F_DP_COST", 24.0))
DP_ARC_EXTRA = float(os.environ.get("P2F_DP_ARC_EXTRA", 8.0))


def resample(contour, n=DP_POINTS):
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, np.concatenate([contour[:, 0], contour[:1, 0]]))
    y = np.interp(t, s, np.concatenate([contour[:, 1], contour[:1, 1]]))
    return np.column_stack([x, y])


class SpanFits:
    """O(1) line- and circle-fit residuals for any span, from prefix sums on a doubled ring."""

    def __init__(self, pts):
        p = np.vstack([pts, pts])
        x, y = p[:, 0], p[:, 1]
        z = x * x + y * y
        cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
        self.p = p
        self.sx, self.sy, self.sz = cs(x), cs(y), cs(z)
        self.sxx, self.syy, self.sxy = cs(x * x), cs(y * y), cs(x * y)
        self.sxz, self.syz, self.szz = cs(x * z), cs(y * z), cs(z * z)

    def line_sse(self, i, j):
        a, b = self.p[i], self.p[j]
        d = b - a
        L = np.hypot(*d)
        if L < 2.0:
            return float("inf")
        nx, ny = -d[1] / L, d[0] / L
        c = -(nx * a[0] + ny * a[1])
        n = j - i + 1
        S = lambda arr: arr[j + 1] - arr[i]
        return max(0.0, nx * nx * S(self.sxx) + 2 * nx * ny * S(self.sxy) + ny * ny * S(self.syy)
                   + 2 * nx * c * S(self.sx) + 2 * ny * c * S(self.sy) + n * c * c)

    def circle_fit(self, i, j):
        n = j - i + 1
        S = lambda arr: arr[j + 1] - arr[i]
        mx, my, mz = S(self.sx) / n, S(self.sy) / n, S(self.sz) / n
        Mxx = S(self.sxx) / n - mx * mx
        Myy = S(self.syy) / n - my * my
        Mxy = S(self.sxy) / n - mx * my
        Mxz = S(self.sxz) / n - mx * mz
        Myz = S(self.syz) / n - my * mz
        det = Mxx * Myy - Mxy * Mxy
        if abs(det) < 1e-12:
            return None
        cx = (Mxz * Myy - Myz * Mxy) / (2 * det)
        cy = (Myz * Mxx - Mxz * Mxy) / (2 * det)
        r2 = cx * cx + cy * cy + mz - 2 * (cx * mx + cy * my)
        if r2 <= 0:
            return None
        ccx, ccy, r = cx + 0, cy + 0, np.sqrt(r2)
        Szz = S(self.szz)
        sse = (Szz - 2 * (cx + 0) * 0)
        # exact SSE of (dist - r): approximate algebraically via variance of z about circle
        # residual estimate: mean((|p-c| - r)^2) ~ mean((z - 2 cx x - 2 cy y - (r^2 - cx^2 - cy^2))^2) / (4 r^2)
        c0 = r2 - cx * cx - cy * cy
        q = (S(self.szz) - 4 * cx * S(self.sxz) - 4 * cy * S(self.syz)
             + 4 * cx * cx * S(self.sxx) + 8 * cx * cy * S(self.sxy) + 4 * cy * cy * S(self.syy)
             - 2 * c0 * S(self.sz) + 4 * c0 * cx * S(self.sx) + 4 * c0 * cy * S(self.sy)
             + n * c0 * c0)
        sse = max(0.0, q / max(4 * r2, 1e-9))
        return ccx, ccy, float(r), float(sse)


def arc_ok(pts_span, cx, cy, r, th):
    v = pts_span - [cx, cy]
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    span = abs(np.degrees(ang[-1] - ang[0]))
    chord = float(np.hypot(*(pts_span[-1] - pts_span[0])))
    cv = pts_span[-1] - pts_span[0]
    sag = float(np.max(np.abs(cv[0] * (pts_span[:, 1] - pts_span[0][1])
                              - cv[1] * (pts_span[:, 0] - pts_span[0][0])) / max(chord, 1e-9)))
    return (th.ARC_MIN_SPAN_DEG < span < th.ARC_MAX_SPAN_DEG
            and sag > th.ARC_MIN_SAG_FRAC * chord)


def decompose(contour, length_px):
    from photo2fcstd import thresholds as th
    pts = resample(np.asarray(contour, float))
    n = len(pts)
    curv = np.abs(np.gradient(np.unwrap(np.arctan2(*np.gradient(pts, axis=0).T[::-1]))))
    start = int(np.argmax(curv))
    ring = np.roll(pts, -start, axis=0)
    fits = SpanFits(ring)
    sig2 = DP_SIGMA * DP_SIGMA
    MIN_SPAN = 4
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    choice = [None] * (n + 1)
    for j in range(MIN_SPAN, n + 1):
        for i in range(max(0, j - n), j - MIN_SPAN + 1):
            if best[i] == np.inf:
                continue
            jj = j
            lc = best[i] + fits.line_sse(i, jj) / sig2 + DP_ELEMENT_COST
            if lc < best[j]:
                best[j] = lc
                choice[j] = (i, "line")
            cf = fits.circle_fit(i, jj)
            if cf is not None:
                cx, cy, r, sse = cf
                span_pts = ring[np.arange(i, jj + 1) % n]
                if arc_ok(span_pts, cx, cy, r, th):
                    ac = best[i] + sse / sig2 + DP_ELEMENT_COST + DP_ARC_EXTRA
                    if ac < best[j]:
                        best[j] = ac
                        choice[j] = (i, "arc")
    if choice[n] is None:
        return None
    cuts = []
    j = n
    while j > 0:
        i, typ = choice[j]
        cuts.append((i, j, typ))
        j = i
    cuts.reverse()
    doubled = np.vstack([ring, ring])
    out = [(doubled[i:j + 1], typ) for i, j, typ in cuts]
    return out, start


def dp_elements(contour, length_px):
    from photo2fcstd.trace import arc_from_run, merge_and_snap, support_of
    got = decompose(contour, length_px)
    if got is None:
        return None
    spans, _ = got
    els = []
    for run, typ in spans:
        if len(run) < 2:
            continue
        if typ == "arc":
            els.append(arc_from_run(run))
        else:
            line = {"type": "line", "p0": run[0].tolist(), "p1": run[-1].tolist(), "_run": run}
            line["support"] = support_of(run, line, length_px)
            els.append(line)
    if len(els) < 2:
        return None
    for e in els:
        if "support" not in e and e.get("_run") is not None:
            e["support"] = support_of(e["_run"], e, length_px)
    return merge_and_snap(els, length_px)
