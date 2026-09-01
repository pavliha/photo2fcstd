"""One page of what today established, in pictures."""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, tilt_model as TM
from photo2fcstd.trace import load, segment_photo
import tilt_close as T
import tilt_synth as TS


def draw_loops(ax, loops, colour="C0"):
    for l in loops:
        if l["type"] == "circle":
            ax.add_patch(Circle((l["cx"], l["cy"]), l["r"], fill=False, lw=1.6, color="C3"))
            continue
        for e in l["elements"]:
            if e["type"] == "line":
                ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.6, color=colour)
            else:
                a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
                a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
                t1, t2 = (a0, a1) if e["ccw"] else (a1, a0)
                ax.add_patch(Arc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"], theta1=t1, theta2=t2,
                                 lw=1.6, color="C2"))
    ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.08); ax.invert_yaxis()


def framed_radius(mesh, normal, start, target=0.55):
    """Put the square-on silhouette across most of the frame, whatever the part's proportions."""
    rad = start
    for _ in range(3):
        v = T.view_at(normal, 0.0, 40.0, rad)
        _, msk = T.shaded(mesh, v)
        if msk.sum() < 50:
            rad *= 1.6
            continue
        ys, xs = np.nonzero(msk)
        span = max(np.ptp(xs), np.ptp(ys))
        rad *= max(span, 1) / (target * T.W)
    return rad


def pick_demo(parts, ideal, tries=24):
    """The part that shows the effect, not the part that happens to be sixth."""
    best = None
    for part in parts[:tries]:
        rec = ideal[part]
        try:
            m = trimesh.load(bench.truth_of(part))
            assert len(m.faces)
        except Exception:
            continue
        m.apply_translation(-m.bounds.mean(0))
        n = T.face_normal(rec)
        rad = framed_radius(m, n, float(np.max(m.extents)) * 4)
        v0 = T.view_at(n, 0.0, 40.0, rad)
        _, m0 = T.shaded(m, v0)
        if m0.sum() < 0.05 * T.W * T.H:
            continue
        def sc(t):
            v = T.view_at(n, t, 40.0, rad)
            _, sol = T.shaded(m, v)
            r = T.score_mask(sol, rec)
            return r["region_iou"] if r else 0.0
        a, b = sc(0.0), sc(15.0)
        if a > 0.85 and (best is None or a - b > best[0]):
            best = (a - b, part, rec)
    return (best[1], best[2]) if best else (parts[5], ideal[parts[5]])


def main(out="docs/figures/tilt.png"):
    ideal = T.IDEAL
    parts = [p for p in sorted(ideal) if SS.trustworthy(ideal[p]) and T.face_normal(ideal[p]) is not None]
    shown = bench.with_photos(parts)[0]
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 4, hspace=0.34, wspace=0.2)

    part = shown[1]
    photo = bench.photos_of(part)[0]
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(load(photo)); ax.set_title("%s: the photograph" % part, fontsize=9)
    mask = np.asarray(segment_photo(photo)) > 0
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(mask, cmap="gray"); ax2.set_title("silhouette (RMBG)", fontsize=9)
    view = analysis.view_from_mask(mask.astype(np.uint8))
    loops = spec_mod.traced_outline(view)
    ax3 = fig.add_subplot(gs[0, 2])
    if loops:
        draw_loops(ax3, loops)
        s = SS.score_one({"outline": {"loops": loops}}, ideal[part])
        v = SS.verdict(s)
        ax3.set_title("the sketch: IoU %.2f, structure %.2f" % (v["region_iou"], v["structure"]), fontsize=9)
    ax4 = fig.add_subplot(gs[0, 3]); ax4.axis("off")
    ax4.text(0, 0.5, "the product is the drawing\n\nregion IoU correlates 0.22 with\nstructural agreement, so both\nare reported now", fontsize=10, va="center")

    demo, rec = pick_demo(parts, ideal)
    n = T.face_normal(rec)
    m = trimesh.load(bench.truth_of(demo)); c = m.bounds.mean(0); m.apply_translation(-c)
    rad = framed_radius(m, n, float(np.max(m.extents)) * 4)
    import cv2
    def face_mask(v):
        img = np.zeros((T.H, T.W), np.uint8)
        for loop in rec["loops"]:
            pts = [e[k] for e in loop for k in ("p0", "p1") if k in e] or [e["c"] for e in loop if "c" in e]
            if len(pts) < 3: continue
            cv2.fillPoly(img, [np.round(T.C.project(np.array(pts, float) - c, v)).astype(np.int32)], 1)
        return img > 0
    def iou(msk):
        s = T.score_mask(msk, rec)
        return s["region_iou"] if s else float("nan")
    panels = []
    for tilt in (0.0, 15.0):
        v = T.view_at(n, tilt, 40.0, rad)
        _, sol = T.shaded(m, v)
        panels.append(("solid, %.0f deg" % tilt, sol, iou(sol)))
    v15 = T.view_at(n, 15.0, 40.0, rad)
    rect = T.rectify(panels[1][1], v15["R"] @ n, T.K_of())
    panels.append(("rectified by the TRUE normal", rect, iou(rect)))
    panels.append(("the base face alone, 15 deg", face_mask(v15), iou(face_mask(v15))))
    union = np.zeros_like(panels[0][1])
    for _, msk, _ in panels:
        union |= msk
    ys, xs = np.nonzero(union)
    pad = int(0.25 * max(np.ptp(xs), np.ptp(ys))) + 8
    box = (max(0, ys.min() - pad), min(T.H, ys.max() + pad),
           max(0, xs.min() - pad), min(T.W, xs.max() + pad))
    for i, (name, msk, sc) in enumerate(panels):
        a = fig.add_subplot(gs[1, i])
        a.imshow(msk[box[0]:box[1], box[2]:box[3]], cmap="gray")
        a.set_title("%s\nIoU %.3f" % (name, sc), fontsize=9)

    rng = np.random.default_rng(3)
    for i in range(2):
        p2 = shown[6 + i]
        a = fig.add_subplot(gs[2, i])
        crop = TM.part_crop(bench.photos_of(p2)[0])
        a.imshow(crop); a.set_title("real photograph %s\n(gate: refused)" % p2, fontsize=9)
    for i in range(2):
        p3 = parts[10 + i]
        mm = trimesh.load(bench.truth_of(p3)); mm.apply_translation(-mm.bounds.mean(0))
        vv = T.view_at(T.face_normal(ideal[p3]), 12.0, 40.0 + 90 * i, float(np.max(mm.extents)) * 4)
        img, mk = TS.lit(mm, vv, rng)
        a = fig.add_subplot(gs[2, 2 + i])
        a.imshow(TS.crop_of(img, mk, rng)); a.set_title("synthetic render %s\n(what training saw)" % p3, fontsize=9)

    d = np.load(os.path.join(ROOT, "data", "tilt_pixels.npz"))
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    X, Y, g = d["obj"], d["normal"], d["part"]
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        for j in range(3):
            pred[te, j] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((X[tr]-mu)/sd, Y[tr, j]).predict((X[te]-mu)/sd)
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    th = np.degrees(np.arccos(np.clip(np.abs(pn[:, 2]), 0, 1)))
    tt = np.degrees(np.arccos(np.clip(np.abs(Y[:, 2]), 0, 1)))
    a = fig.add_subplot(gs[3, 0:2])
    a.scatter(tt, th, s=4, alpha=0.25, color="C0")
    a.plot([0, 90], [0, 90], "k--", lw=1)
    a.axvspan(0, 30, color="C2", alpha=0.08)
    a.set_xlabel("true tilt, degrees"); a.set_ylabel("predicted")
    a.set_title("T-LESS photographs, held out by object\n2.63 deg error below 30 deg (shaded), 87%% within 5", fontsize=9)
    a.set_aspect("equal"); a.set_xlim(0, 90); a.set_ylim(0, 90)

    a = fig.add_subplot(gs[3, 2:4]); a.axis("off")
    a.text(0, 1.0, "what today established", fontsize=12, fontweight="bold", va="top")
    a.text(0, 0.88,
           "tilt IS in the pixels\n"
           "    4.8 deg full normal, 2.63 deg tilt below 30 deg\n"
           "    shading beats the silhouette by 3.6 deg [3.4, 3.9]\n"
           "    background contributes nothing (+0.05 deg)\n\n"
           "but it CANNOT be corrected\n"
           "    projective distortion at 15 deg: -0.009 [-0.026, +0.008]\n"
           "    side walls appearing:            -0.084 [-0.114, -0.058]\n"
           "    rectifying by the true normal:   -0.013\n\n"
           "and renders cannot train it\n"
           "    photographs 2.63 deg, renders 7.6 to 8.0\n"
           "    against a constant of 6.98 and 8.81\n"
           "    two renderers tried, the better one slightly worse\n\n"
           "so the check ships gated, and abstains honestly.\n"
           "docs/tilt-labels.md is the afternoon that finishes it.",
           fontsize=9.5, va="top", family="monospace")

    for a in fig.get_axes():
        if a.get_title() or a.images:
            if a.images:
                a.set_xticks([]); a.set_yticks([])
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    fig.suptitle("photo2fcstd - tilt: measurable, not correctable", fontsize=14)
    plt.savefig(os.path.join(ROOT, out), dpi=72, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
