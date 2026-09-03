"""Render the E1 survey pairs to PNGs plus a key mapping sides back to arms."""
import json, os, random, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def draw_spec(ax, row):
    rv, ol = row.get("revolve"), row.get("outline")
    if rv:
        P = np.array(rv["profile"] + rv["profile"][:1])
        ax.plot(P[:, 0], P[:, 1], "-", lw=1.6, color="#1a1a1a")
        ax.plot(-P[:, 0], P[:, 1], "-", lw=1.6, color="#1a1a1a", alpha=0.25)
        ax.axvline(0, color="#1a1a1a", ls="--", lw=0.7, alpha=0.5)
        for h in rv.get("holes", []):
            if h["type"] == "circle":
                ax.add_patch(Circle((h["cx"], h["cy"]), h["r"], fill=False, lw=1.4, color="#1a1a1a"))
        return
    for l in ol["loops"]:
        if l["type"] == "circle":
            ax.add_patch(Circle((l["cx"], l["cy"]), l["r"], fill=False, lw=1.6, color="#1a1a1a"))
            continue
        if l["type"] == "ellipse":
            ax.add_patch(Ellipse((l["cx"], l["cy"]), 2 * l["a"], 2 * l["b"],
                                 angle=np.degrees(l["theta"]), fill=False, lw=1.6, color="#1a1a1a"))
            continue
        for e in l["elements"]:
            if e["type"] == "line":
                ax.plot([e["p0"][0], e["p1"][0]], [e["p0"][1], e["p1"][1]], "-", lw=1.6, color="#1a1a1a",
                        solid_capstyle="round")
            elif e["type"] == "arc":
                a0 = np.degrees(np.arctan2(e["p0"][1] - e["cy"], e["p0"][0] - e["cx"]))
                a1 = np.degrees(np.arctan2(e["p1"][1] - e["cy"], e["p1"][0] - e["cx"]))
                t1, t2 = (a0, a1) if e.get("ccw") else (a1, a0)
                ax.add_patch(Arc((e["cx"], e["cy"]), 2 * e["r"], 2 * e["r"], theta1=t1, theta2=t2,
                                 lw=1.6, color="#1a1a1a"))


def counts_of(row):
    from photo2fcstd.sketch_score import counts_of_spec
    return counts_of_spec({"outline": row.get("outline"), "revolve": row.get("revolve")})


def main():
    from photo2fcstd import bench
    from photo2fcstd.trace import load
    pairs = json.load(open(os.path.join(ROOT, "data", "pair_survey.json")))
    outdir = os.path.join(ROOT, "data", "pair_survey_png")
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(11)
    key = []
    for i, p in enumerate(pairs, start=1):
        flip = rng.random() < 0.5
        left, right = ("variant", "shipped") if flip else ("shipped", "variant")
        src = (p["shipped"].get("outline") or p["shipped"].get("revolve") or {}).get("source")
        if not src:
            src = bench.photos_of(p["part"])[0]
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
        ax[0].imshow(load(src))
        ax[0].set_title("photo", fontsize=9)
        for a, side, label in ((ax[1], left, "A"), (ax[2], right, "B")):
            draw_spec(a, p[side])
            c = counts_of(p[side])
            a.set_title("%s   (%s)" % (label, ", ".join("%d %s" % (n, k) for k, n in sorted(c.items()))),
                        fontsize=9)
            a.set_aspect("equal"); a.invert_yaxis()
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        fig.suptitle("pair %d" % i, fontsize=11, x=0.05, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fn = os.path.join(outdir, "pair%02d.png" % i)
        fig.savefig(fn, dpi=80); plt.close(fig)
        key.append({"pair": i, "part": p["part"], "knob": p["arm"], "A": left, "B": right,
                    "verdicts": {"A": p[left]["verdict"], "B": p[right]["verdict"]}})
    json.dump(key, open(os.path.join(ROOT, "data", "pair_survey_key.json"), "w"), indent=1)
    print("wrote %d pairs to %s and the key" % (len(key), outdir))


if __name__ == "__main__":
    main()
