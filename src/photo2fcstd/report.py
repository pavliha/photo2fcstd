import argparse
import concurrent.futures
import glob
import html
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from photo2fcstd import overlay
from photo2fcstd.gallery import draw_sketch
from photo2fcstd.settings import data_dir

PHOTOS = os.path.join(data_dir(), "captured_img")


def photo_path(name):
    hits = glob.glob(os.path.join(PHOTOS, "*", os.path.basename(name)))
    return hits[0] if hits else None


def results(run):
    rows = (l.split() for l in open(os.path.join(run, "results.txt")))
    return {f[0]: (f[1], float(f[2])) for f in rows if len(f) == 3 and f[2] != "fail"}


def failures(run):
    rows = (l.split() for l in open(os.path.join(run, "results.txt")))
    return {f[0]: f[1] for f in rows if len(f) == 3 and f[2] == "fail"}


def spec_of(run, part):
    path = os.path.join(run, "out", part + ".spec.json")
    return json.load(open(path)) if os.path.exists(path) else None


def chosen(spec):
    if not spec:
        return None
    src = spec.get("source") or (spec.get("revolve") or spec.get("outline") or {}).get("source")
    if src:
        return os.path.basename(src)
    views = spec.get("views") or {}
    front = views.get("front") or {}
    return os.path.basename(front["source"]) if front.get("source") else None


def truth_path(part):
    return os.path.join(data_dir(), "stl_from_step", part + ".stl")


def photo_strip(part, pick):
    shots = sorted(glob.glob(os.path.join(PHOTOS, "*", part + "_?.jpg")))[:3]
    if not shots:
        return None, []
    images = [ImageOps.exif_transpose(Image.open(p)) for p in shots]
    h = min(im.height for im in images)
    scaled = [im.resize((int(im.width * h / im.height), h)) for im in images]
    pad = max(6, h // 90)
    sheet = Image.new("RGB", (sum(im.width for im in scaled) + pad * (len(scaled) - 1), h), "#faf8f5")
    edges, x = [], 0
    for im, path in zip(scaled, shots):
        sheet.paste(im, (x, 0))
        edges.append((x, im.width, os.path.basename(path) == pick))
        x += im.width + pad
    return sheet, edges


def draw_trace(ax, source):
    from photo2fcstd.trace import outline, segment_photo, upright_mask
    mask, _ = upright_mask(segment_photo(source))
    poly, shape = outline(mask)
    ax.imshow(mask, cmap="gray")
    ax.plot(*np.vstack([poly, poly[:1]]).T, "-", lw=1.2, color="#d43d2a")
    for hole in shape["holes"]:
        h = np.array(hole)
        ax.plot(*np.vstack([h, h[:1]]).T, "-", lw=1.0, color="#2aa6d4")
    ax.set_title("cutout + trace: %d pts, %d holes" % (len(poly), len(shape["holes"])), fontsize=8)


def card(args):
    run, part, out = args
    if os.path.exists(out):
        return part, True
    spec = spec_of(run, part)
    stl = os.path.join(run, "out", part + ".stl")
    if spec is None or not os.path.exists(stl):
        return part, False
    try:
        fig = plt.figure(figsize=(17, 3.5))
        gs = fig.add_gridspec(1, 5, width_ratios=(2.9, 1.15, 1.15, 1, 1), wspace=0.12)
        ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
        pick = chosen(spec)
        sheet, edges = photo_strip(part, pick)
        if sheet is not None:
            ax[0].imshow(sheet)
            for x, w, is_pick in edges:
                ax[0].add_patch(Rectangle((x, 0), w - 1, sheet.height - 1, fill=False,
                                          lw=3 if is_pick else 1,
                                          edgecolor="#2f8f4e" if is_pick else "#d8d2c8"))
            ax[0].set_title("three photos - traced: %s" % (pick or "?"), fontsize=8)
        source = photo_path(pick) if pick else None
        draw_trace(ax[1], source) if source else ax[1].set_title("cutout unavailable", fontsize=8)
        draw_sketch(ax[2], spec)
        c = overlay.compare(truth_path(part), stl)
        overlay.draw(ax[3], c["views"]["face"], "face")
        overlay.draw(ax[4], c["views"]["side"], "side")
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
            a.axis("off")
        fig.suptitle("%s   %s   3D IoU %.3f   missing %.0f%%  extra %.0f%%"
                     % (part, spec.get("mode", "?"), c["iou3d"], 100 * c["missing3d"], 100 * c["extra3d"]),
                     fontsize=10, y=0.985)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)
        fig.savefig(out, dpi=62, facecolor="#faf8f5")
        plt.close(fig)
        return part, True
    except Exception:
        plt.close("all")
        return part, False


def render(run, parts, jobs):
    thumbs = os.path.join(run, "thumbs")
    os.makedirs(thumbs, exist_ok=True)
    work = [(run, p, os.path.join(thumbs, p + ".png")) for p in parts]
    todo = [w for w in work if not os.path.exists(w[2])]
    done = {}
    if todo:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, (part, ok) in enumerate(pool.map(card, todo, chunksize=4)):
                done[part] = ok
                if (i + 1) % 25 == 0:
                    print("  rendered %d/%d" % (i + 1, len(todo)), file=sys.stderr)
    return {p: os.path.exists(os.path.join(thumbs, p + ".png")) for p in parts}


def summary(run):
    path = os.path.join(run, "summary.txt")
    return open(path).read().strip() if os.path.exists(path) else ""


def rows_for(run, base):
    mine, theirs = results(run), (results(base) if base else {})
    bad = failures(run)
    out = []
    for part, (mode, iou) in sorted(mine.items()):
        spec, other = spec_of(run, part), (spec_of(base, part) if base else None)
        prior = theirs.get(part)
        out.append({"part": part, "mode": mode, "iou": iou,
                    "delta": (iou - prior[1]) if prior else None,
                    "view": chosen(spec), "base_view": chosen(other) if other else None,
                    "base_mode": prior[0] if prior else None, "failed": False})
    out += [{"part": p, "mode": m, "iou": None, "delta": None, "view": None,
             "base_view": None, "base_mode": None, "failed": True} for p, m in sorted(bad.items())]
    return out


TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>%(title)s</title>
<style>
:root{--bg:#faf8f5;--fg:#241f1a;--mut:#8a8177;--line:#e2dbd1;--card:#fff;--good:#2f8f4e;--bad:#b0522f}
@media (prefers-color-scheme:dark){:root{--bg:#16130f;--fg:#eee7dd;--mut:#9a9088;--line:#332c25;--card:#1e1a15}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 20px;z-index:5}
h1{margin:0 0 6px;font-size:17px;font-weight:650}
pre{margin:6px 0 0;color:var(--mut);font-size:12px;white-space:pre-wrap}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
input,select,button{font:inherit;padding:5px 9px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg)}
main{padding:16px 20px;display:flex;flex-direction:column;gap:14px}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.head{display:flex;gap:14px;align-items:baseline;padding:9px 13px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.part{font-weight:650;font-variant-numeric:tabular-nums}
.num{font-variant-numeric:tabular-nums}
.up{color:var(--good)}.down{color:var(--bad)}
.tag{color:var(--mut);font-size:12px}
img{width:100%%;display:block}
.miss{padding:12px 13px;color:var(--mut);font-size:13px}
footer{padding:20px;color:var(--mut);font-size:12px}
</style>
<header>
<h1>%(title)s</h1>
<pre>%(summary)s</pre>
<div class="bar">
<input id="q" placeholder="part id" size="10">
<select id="mode"><option value="">all modes</option>%(modes)s</select>
<select id="sort">
<option value="part">part</option>
<option value="iou">worst IoU first</option>
<option value="iou-desc">best IoU first</option>
%(deltasort)s
</select>
<label class="tag"><input type="checkbox" id="changed"> only where the traced photo changed</label>
<span class="tag" id="count"></span>
</div>
</header>
<main id="list"></main>
<footer>Grey = both, red = ground truth only, blue = model only. Generated by photo2fcstd-report.</footer>
<script>
const DATA = %(data)s, HAS_BASE = %(hasbase)s;
const list = document.getElementById("list"), count = document.getElementById("count");
const fmt = v => v === null ? "-" : v.toFixed(3);
const render = () => {
  const q = document.getElementById("q").value.trim();
  const mode = document.getElementById("mode").value;
  const onlyChanged = document.getElementById("changed").checked;
  const sort = document.getElementById("sort").value;
  const rows = DATA
    .filter(r => !q || r.part.includes(q))
    .filter(r => !mode || r.mode === mode)
    .filter(r => !onlyChanged || (r.base_view && r.view !== r.base_view))
    .sort((a, b) => sort === "part" ? a.part.localeCompare(b.part)
      : sort === "iou" ? (a.iou ?? -1) - (b.iou ?? -1)
      : sort === "iou-desc" ? (b.iou ?? -1) - (a.iou ?? -1)
      : sort === "delta" ? (a.delta ?? 0) - (b.delta ?? 0)
      : (b.delta ?? 0) - (a.delta ?? 0));
  count.textContent = rows.length + " of " + DATA.length + " parts";
  list.innerHTML = rows.map(r => {
    const d = r.delta === null ? "" :
      `<span class="num ${r.delta > 0.005 ? "up" : r.delta < -0.005 ? "down" : ""}">${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(3)}</span>`;
    const moved = r.base_view && r.view !== r.base_view
      ? `<span class="tag">traced ${r.base_view} &rarr; ${r.view}</span>` : "";
    const body = r.failed
      ? `<div class="miss">build failed</div>`
      : `<img loading="lazy" src="thumbs/${r.part}.png" alt="${r.part}"`
        + ` onerror="this.replaceWith(Object.assign(document.createElement('div'),`
        + `{className:'miss',textContent:'still rendering - reload the page'}))">`;
    return `<div class="row"><div class="head"><span class="part">${r.part}</span>`
      + `<span class="tag">${r.mode}</span><span class="num">IoU ${fmt(r.iou)}</span>${d}${moved}</div>${body}</div>`;
  }).join("");
};
["q", "mode", "sort", "changed"].forEach(id =>
  document.getElementById(id).addEventListener("input", render));
render();
</script>
"""


def page_for(run, base, rows):
    modes = sorted({r["mode"] for r in rows})
    title = "photo2fcstd - %s%s" % (os.path.basename(run.rstrip("/")),
                                    (" vs %s" % os.path.basename(base.rstrip("/"))) if base else "")
    page = TEMPLATE % {
        "title": html.escape(title),
        "summary": html.escape(summary(run)),
        "modes": "".join("<option>%s</option>" % html.escape(m) for m in modes),
        "deltasort": '<option value="delta">biggest regression first</option><option value="delta-desc">biggest gain first</option>' if base else "",
        "data": json.dumps(rows),
        "hasbase": "true" if base else "false",
    }
    return page


def build(run, base, limit, jobs, out):
    rows = rows_for(run, base)
    open(out, "w").write(page_for(run, base, rows))
    ordered = sorted(rows, key=lambda r: abs(r["delta"] or 0), reverse=True) if base else rows
    keep = ordered[:limit] if limit else ordered
    ok = render(run, [r["part"] for r in keep], jobs)
    open(out, "w").write(page_for(run, base, rows))
    return out, sum(1 for r in rows if ok.get(r["part"])), len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="photo2fcstd-report")
    p.add_argument("run")
    p.add_argument("--baseline")
    p.add_argument("--limit", type=int)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out")
    a = p.parse_args(argv)
    out = a.out or os.path.join(a.run, "report.html")
    path, shown, total = build(a.run, a.baseline, a.limit, a.jobs, out)
    print("%s - %d of %d parts have a thumbnail" % (path, shown, total))


def run():
    main()


if __name__ == "__main__":
    main()
