"""Draw the board capture path: rendered photos, recovered pose, carved volume, sketch."""
import json, os, sys

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2fcstd import capture_check as CK, carve as C, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def draw_rings(ax, rings, color, title):
    for r in rings:
        r = np.asarray(r, float)
        ax.plot(*np.vstack([r, r[:1]]).T, "-", lw=1.8, color=color)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.8)


def main(part="00964"):
    w_mm, h_mm = CK.board_mm()
    mesh = trimesh.load(truth_of(part))
    mesh.apply_translation(-mesh.bounds[0])
    ext = mesh.extents.copy()
    mesh.apply_translation([w_mm / 2 - ext[0] / 2, h_mm / 2 - ext[1] / 2, 0.0])
    views = CK.board_views(16, radius=max(w_mm, h_mm) * 1.6, elevations=CK.DETECTABLE_ELEV)
    rows = CK.recover(views, mesh=mesh)
    found = [r for r in rows if r["got"]]

    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1])
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        img, _ = CK.render_view(views[i * 4], CK.board_texture(), mesh)
        ax.imshow(img[:, :, ::-1])
        ax.set_xticks([]); ax.set_yticks([])
        r = rows[i * 4]
        ax.set_title("view %d   pose off by %.3f deg, %.2f mm" % (i * 4 + 1, r["deg"], r["mm"]),
                     fontsize=8)
    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    deg = np.array([r["deg"] for r in found])
    ax.text(0.0, 0.95, "%d of %d views solved" % (len(found), len(rows)), fontsize=11,
            fontweight="bold", va="top")
    ax.text(0.0, 0.78, "rotation  %.3f deg median\nposition  %.3f mm median"
            % (np.median(deg), np.median([r["mm"] for r in found])), fontsize=10, va="top",
            family="monospace")
    ax.text(0.0, 0.52, "the pose budget is 2 deg\nbefore carving falls behind\na single photo",
            fontsize=10, va="top", color="#1a7f37")
    ax.text(0.0, 0.24, "part %s\n%d x %d x %d mm" % (part, *np.round(ext).astype(int)),
            fontsize=10, va="top", family="monospace", color="#555")

    carved = C.carve([r["got"] for r in found], [r["mask"] for r in found], voxel_mm=0.4,
                     bounds=((w_mm / 2 - 30, w_mm / 2 + 30), (h_mm / 2 - 30, h_mm / 2 + 30),
                             (0.0, float(ext[2]) * 2 + 10)))
    carved["min_elevation_deg"] = min(C.view_elevation_deg(r["got"]) for r in found)
    fixed = C.debias_height(carved)

    for j, (cv, label) in enumerate(((carved, "carved hull"), (fixed, "after height debias"))):
        ax = fig.add_subplot(gs[1, j])
        pts = cv["points_mm"]
        ax.scatter(pts[:, 0], pts[:, 2], s=0.4, c="#333", marker=".")
        ax.axhline(ext[2], color="#1a7f37", ls="--", lw=1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s\nheight %.1f mm, true %.1f (green)"
                     % (label, np.ptp(pts[:, 2]), ext[2]), fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(C.occupancy(fixed, axis=C.base_axis(fixed)), cmap="gray_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("looking down the learned axis", fontsize=9)

    spec = C.spec_from_carve(fixed, name=part)
    draw_rings(fig.add_subplot(gs[1, 3]), SS.spec_rings(spec), "#1a7f37",
               "the sketch, IoU %.2f" % SS.score_one(spec, IDEAL[part])["region_iou"])
    draw_rings(fig.add_subplot(gs[1, 4]), SS.ideal_rings(IDEAL[part]), "#111",
               "the real sketch (STEP)")

    fig.suptitle("Photos of a part on a printed target, in: a parametric sketch, out", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(ROOT, "docs", "figures", "capture_path.png")
    plt.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
