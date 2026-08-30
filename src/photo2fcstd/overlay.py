import sys

import numpy as np
import trimesh

from photo2fcstd.score import FINE, best_alignment




def grid(vox, pitch=FINE):
    N = int(round(1 / pitch)) + 2
    g = np.zeros((N, N, N), bool)
    idx = (np.array(list(vox)) if vox else np.zeros((0, 3), int)) + N // 2
    if len(idx) and (idx.min() < 0 or idx.max() >= N):
        raise ValueError("voxel index out of grid: %d..%d for N=%d" % (idx.min(), idx.max(), N))
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return g


def compare(ref_path, cand_path, pitch=FINE):
    ref_mesh, cand_mesh = trimesh.load(ref_path), trimesh.load(cand_path)
    iou3, raw, R, ref_v, cand_v = best_alignment(ref_mesh, cand_mesh, pitch)
    a, b = grid(ref_v, pitch), grid(cand_v, pitch)
    occ = np.argwhere(a)
    order = np.argsort(occ.max(axis=0) - occ.min(axis=0))[::-1]
    views = {}
    for name, axis in (("face", order[2]), ("side", order[1])):
        pa, pb = a.any(axis=axis), b.any(axis=axis)
        inter, union = (pa & pb).sum(), (pa | pb).sum()
        views[name] = {"ref": pa, "cand": pb, "iou": inter / max(union, 1), "missing": (pa & ~pb).sum() / max(pa.sum(), 1), "extra": (pb & ~pa).sum() / max(pa.sum(), 1)}
    return {"iou3d": iou3, "views": views, "missing3d": len(ref_v - cand_v) / max(len(ref_v), 1), "extra3d": len(cand_v - ref_v) / max(len(ref_v), 1)}


def draw(ax, v, title):
    img = np.ones(v["ref"].shape + (3,))
    img[v["ref"] & v["cand"]] = (0.55, 0.55, 0.55)
    img[v["ref"] & ~v["cand"]] = (0.85, 0.15, 0.15)
    img[~v["ref"] & v["cand"]] = (0.15, 0.35, 0.85)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_title("%s  IoU %.2f   missing %.0f%%  extra %.0f%%" % (title, v["iou"], 100 * v["missing"], 100 * v["extra"]), fontsize=8)
    ax.axis("off")


def panel(pairs, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(len(pairs), 3, figsize=(11, 3.4 * len(pairs)), squeeze=False)
    for r, (label, photo, ref, cand) in enumerate(pairs):
        c = compare(ref, cand)
        if photo:
            from PIL import Image, ImageOps
            ax[r, 0].imshow(ImageOps.exif_transpose(Image.open(photo)))
        ax[r, 0].set_title("%s   3D IoU %.2f  (missing %.0f%%, extra %.0f%%)" % (label, c["iou3d"], 100 * c["missing3d"], 100 * c["extra3d"]), fontsize=9, fontweight="bold", loc="left")
        ax[r, 0].axis("off")
        draw(ax[r, 1], c["views"]["face"], "face view")
        draw(ax[r, 2], c["views"]["side"], "side view")
    fig.legend(handles=[Patch(color=(0.55, 0.55, 0.55), label="both"), Patch(color=(0.85, 0.15, 0.15), label="ground truth only (missing)"), Patch(color=(0.15, 0.35, 0.85), label="model only (extra)")], loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.savefig(out, dpi=70)


if __name__ == "__main__":
    c = compare(sys.argv[1], sys.argv[2])
    print("3D IoU %.3f  missing %.0f%%  extra %.0f%%   face-view IoU %.2f   side-view IoU %.2f" % (c["iou3d"], 100 * c["missing3d"], 100 * c["extra3d"], c["views"]["face"]["iou"], c["views"]["side"]["iou"]))
