"""What the set predictor draws, against the tracer, on sketches neither has seen."""
import os, sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc as MplArc, Circle as MplCircle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402
from photo2fcstd import sketchnet as SN  # noqa: E402

INK, NET, TRACE = "#111", "#1a5fb4", "#b32d2e"


def draw(ax, els, colour, title):
    for e in els:
        if e["type"] == "circle":
            ax.add_patch(MplCircle((e["cx"], e["cy"]), e["r"], fill=False, lw=1.8, color=colour))
        elif e["type"] == "arc":
            a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
            a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
            t1, t2 = (a0, a1) if e.get("ccw", True) else (a1, a0)
            ax.add_patch(MplArc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"],
                                theta1=t1, theta2=t2, lw=1.8, color=colour))
        else:
            ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.8, color=colour)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 768); ax.set_ylim(768, 0)
    ax.set_title(title, fontsize=9, color=colour, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(1.7)


def tracer_on(filled):
    import cv2
    from photo2fcstd.trace import outline, primitives
    big = cv2.resize(filled, (768, 768), interpolation=cv2.INTER_NEAREST) > 0.5
    _, shape = outline(big)
    loops = primitives([shape["raw"]] + shape["raw_holes"], 768.0)
    els = []
    for l in loops:
        if l["type"] == "circle":
            els.append({"type": "circle", "cx": l["cx"], "cy": l["cy"], "r": l["r"]})
        else:
            els += l["elements"]
    return els


def main(n=5):
    stroke = np.load(os.path.join(ROOT, "data", "sketchnet_tilt0.npz"), allow_pickle=True)
    filled = np.load(os.path.join(ROOT, "data", "sketchnet_filled.npz"), allow_pickle=True)
    parts = stroke["parts"]
    uniq = sorted(set(parts.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[int(0.8 * len(uniq)):])
    te = [i for i, p in enumerate(parts) if p in held]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SN.build_model().to(dev)
    m.load_state_dict(torch.load(SN.MODEL_PATH + ".pretrained", map_location=dev))
    m.eval()
    box = (np.zeros(2), 768.0)
    picks, seen = [], set()
    for i in te:
        truth = SN.decode(stroke["Y"][i], box)
        if 4 <= len(truth) <= 12 and parts[i] not in seen:
            picks.append(i); seen.add(parts[i])
        if len(picks) >= n:
            break
    fig, ax = plt.subplots(len(picks), 4, figsize=(13.5, 3.35 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        with torch.no_grad():
            raw = m(torch.from_numpy(stroke["X"][i]).float()[None, None].to(dev)).cpu().numpy()[0]
        raw[:, 0] = 1 / (1 + np.exp(-raw[:, 0]))
        pred = SN.decode(raw, box)
        truth = SN.decode(stroke["Y"][i], box)
        tr = tracer_on(filled["X"][i])
        ax[r, 0].imshow(stroke["X"][i], cmap="gray_r")
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        ax[r, 0].set_title("the raster it is given", fontsize=9)
        ax[r, 0].set_ylabel(str(parts[i]), fontsize=10, fontweight="bold")
        draw(ax[r, 1], tr, TRACE, "geometric tracer: %d primitives" % len(tr))
        draw(ax[r, 2], pred, NET, "sketchnet: %d primitives" % len(pred))
        draw(ax[r, 3], truth, INK, "the real sketch: %d" % len(truth))
    fig.suptitle("Pretrained on 60000 SketchGraphs sketches, tested on PrintCAD parts it has never seen\n"
                 "right primitive count: tracer 34%, sketchnet 46%", fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(ROOT, "docs", "figures", "sketchnet.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
