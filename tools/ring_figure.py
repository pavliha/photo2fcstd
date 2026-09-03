"""01407: why a perfect drawing scores 0.557, in pictures."""
import glob, json, os, sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PIL import Image as _Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd.trace import load, largest, segment_photo  # noqa: E402

MODELS = (("RMBG-2.0", "briaai/RMBG-2.0", 1024),
          ("BiRefNet", "ZhengPeng7/BiRefNet", 1024),
          ("BiRefNet_HR", "ZhengPeng7/BiRefNet_HR", 2048))


def alpha_of(im, name, size, dev):
    from transformers import AutoModelForImageSegmentation
    m = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True)
    m = m.to(torch.float32).to(dev).eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        out = m(tf(im).unsqueeze(0).to(dev).float())
    pred = (out[-1] if isinstance(out, (list, tuple)) else out).sigmoid().cpu().float()
    pred = pred[0, 0] if pred.ndim == 4 else pred[0]
    return largest(np.asarray(transforms.ToPILImage()(pred).resize(im.size)) > 128)


def circles(mask):
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    ar = [cv2.contourArea(x) for x in c]
    o = np.argsort(ar)[::-1]
    (ox, oy), ro = cv2.minEnclosingCircle(c[o[0]])
    (hx, hy), rh = cv2.minEnclosingCircle(c[o[1]])
    return (ox, oy, ro), (hx, hy, rh)


def main(out="docs/figures/ring.png"):
    path = glob.glob(os.path.join(os.path.expanduser("~/3DPrint/tools/data/printcad/PrintCAD"),
                                  "captured_img", "*", "01407_2.jpg"))[0]
    img = load(path)
    im = _Image.fromarray((img * 255).astype(np.uint8))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = os.path.join(ROOT, "data", "ring_matting.json")
    got = json.load(open(cache)) if os.path.exists(cache) else {}
    for label, name, size in MODELS:
        if label not in got:
            got[label] = circles(alpha_of(im, name, size, dev))
            json.dump(got, open(cache, "w"))
        print("  %s ratio %.3f" % (label, got[label][1][2] / got[label][0][2]))

    (ox, oy, ro), (hx, hy, rh) = got["RMBG-2.0"]
    ideal_rh = ro * 0.98

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.22)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    ax.add_patch(Circle((ox, oy), ro, fill=False, color="C0", lw=1.6, label="outer, measured"))
    ax.add_patch(Circle((hx, hy), rh, fill=False, color="C3", lw=1.6, label="hole, measured"))
    ax.add_patch(Circle((ox, oy), ideal_rh, fill=False, color="C2", lw=1.6, ls="--", label="hole the STEP wants"))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("01407 as photographed\nthe drawing is two circles, and both are the right primitive", fontsize=9)

    pad = 90
    x0, x1 = int(ox - ro - pad), int(ox - ro + pad)
    y0, y1 = int(oy - pad), int(oy + pad)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img[y0:y1, x0:x1])
    for r, col, ls, lab in ((ro, "C0", "-", "outer"), (rh, "C3", "-", "hole measured"),
                            (ideal_rh, "C2", "--", "hole wanted")):
        ax.add_patch(Circle((ox - x0, oy - y0), r, fill=False, color=col, lw=1.8, ls=ls, label=lab))
    ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("the wall, close up: 29.7 px measured, 16.8 px specified", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    names = [m[0] for m in MODELS]
    ratios = [got[n][1][2] / got[n][0][2] for n in names]
    ax.barh(names, ratios, color="C0", height=0.5)
    ax.axvline(0.980, color="C2", ls="--", lw=2)
    ax.annotate("what the STEP file wants", (0.980, 2.35), (0.9705, 2.55), color="C2", fontsize=8,
                ha="right", arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlim(0.94, 0.99)
    for i, r in enumerate(ratios):
        ax.text(r - 0.0015, i, "%.3f" % r, ha="right", va="center", color="white", fontsize=9)
    ax.set_xlabel("hole radius / outer radius")
    ax.set_title("three matting models, two architectures, two resolutions", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    walls = []
    for r in IDEAL.values():
        rs = sorted(e["r"] for loop in r.get("loops", []) or [] for e in loop if e.get("type") == "circle")
        gaps = [b - a for a, b in zip(rs, rs[1:])]
        if gaps and min(gaps) > 0:
            walls.append(min(gaps))
    w = np.array(walls)
    n, _, _ = ax.hist(w[w < 5], bins=40, color="C0")
    top = n.max()
    ax.set_ylim(0, top * 1.42)
    ax.axvline(0.4, color="C3", ls="--", lw=2)
    ax.axvline(0.2, color="C2", ls="--", lw=2)
    ax.annotate("a 0.4 mm nozzle", (0.4, top * 1.06), (1.15, top * 1.06), color="C3", fontsize=8,
                va="center", arrowprops=dict(arrowstyle="->", color="C3", lw=1))
    ax.annotate("01407 asks for 0.20", (0.2, top * 1.26), (1.15, top * 1.26), color="C2", fontsize=8,
                va="center", arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    ax.set_xlabel("specified wall between concentric circles, mm")
    ax.set_title("10%% of %d parts specify a wall no printer can lay" % len(walls), fontsize=9)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 1.0, "every step pointed elsewhere", fontsize=12, fontweight="bold", va="top")
    ax.text(0, 0.9,
            "the drawing .... 2 circles, exact, structure 1.000\n"
            "the score ...... region IoU 0.557, extra 0.747\n\n"
            "not the mode ... plan and revolve both 0.557\n"
            "not the rule ... its ideal IS two circles\n"
            "not the tracer . follows the mask to 0.001\n"
            "not the mask ... 3 models agree within 0.5 px\n"
            "not the matte .. resolution and threshold\n"
            "                 both flat\n"
            "not the bore ... ratio same across 21-43 deg\n"
            "                 of tilt\n\n"
            "the part ....... 0.20 mm wall specified,\n"
            "                 0.35 mm printed, 0.4 mm nozzle\n\n"
            "a low score here measures the printer.",
            fontsize=9.5, va="top", family="monospace")

    fig.suptitle("photo2fcstd - a perfect drawing scoring 0.557, and why nothing downstream can fix it", fontsize=13)
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    plt.savefig(os.path.join(ROOT, out), dpi=70, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
