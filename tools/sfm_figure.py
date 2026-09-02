"""What the SfM gate actually produced, in pictures."""
import json, os, shutil, sys, tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sfm_check as S
from photo2fcstd import bench, carve as C, sketch_score as SS


def main(part="00005", views=16, voxel=0.6, out="docs/figures/sfm.png"):
    rng = np.random.default_rng(0)
    mesh = trimesh.load(bench.truth_of(part))
    mesh.apply_translation(-np.array([mesh.bounds[:, 0].mean(), mesh.bounds[:, 1].mean(), mesh.bounds[0][2]]))
    radius = float(np.max(mesh.extents)) * 1.9
    truth = S.poses(views, radius)
    lo, hi = 30, 225
    texture = cv2.GaussianBlur(rng.integers(lo, hi, (600, 600, 3), dtype=np.uint8), (0, 0), 1.4)
    work = tempfile.mkdtemp(prefix="sfmfig_")
    images = os.path.join(work, "images")
    os.makedirs(images, exist_ok=True)
    shots, masks = [], []
    for i, v in enumerate(truth):
        img, m = S.render(mesh, v, texture, float(np.max(mesh.extents)) * 1.6, rng)
        cv2.imwrite(os.path.join(images, "v%03d.png" % i), img)
        shots.append(img)
        masks.append(m)
    model, err = S.run_colmap(images, work)
    if model is None:
        print("SfM failed:", err)
        return
    got = S.read_images(model)
    names = sorted(got)
    idx = [int(n[1:4]) for n in names]
    sfm_c = np.array([-got[n][0].T @ got[n][1] for n in names])
    true_c = np.array([truth[i]["eye"] for i in idx])
    scale, Rw, tw = S.umeyama(sfm_c, true_c)
    aligned = (scale * (Rw @ sfm_c.T).T) + tw
    rot = []
    est_views = []
    for k, (n, i) in enumerate(zip(names, idx)):
        R_est = got[n][0] @ Rw.T
        d = R_est @ truth[i]["R"].T
        rot.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
        est_views.append({"rvec": cv2.Rodrigues(R_est)[0], "tvec": (-R_est @ aligned[k]).reshape(3, 1),
                          "K": S.K_of(), "dist": np.zeros(5)})
    bounds = [(float(mesh.bounds[0][k]) - 2, float(mesh.bounds[1][k]) + 2) for k in range(2)] \
        + [(0.0, float(mesh.bounds[1][2]) + 2)]
    a = C.carve([truth[i] for i in idx], [masks[i] for i in idx], voxel_mm=voxel, bounds=bounds)
    b = C.carve(est_views, [masks[i] for i in idx], voxel_mm=voxel, bounds=bounds)
    A = set(map(tuple, np.round(a["points_mm"] / voxel).astype(int)))
    B = set(map(tuple, np.round(b["points_mm"] / voxel).astype(int)))
    agree = len(A & B) / max(len(A | B), 1)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.24)
    for j in range(2):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(cv2.cvtColor(shots[j * 5], cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("what SfM sees: view %d" % (j * 5), fontsize=9)

    fig.text(0.13, 0.895, "the part is textureless - the patterned mat carries every feature SfM uses",
             fontsize=9, style="italic", color=(0.3, 0.3, 0.3))
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(true_c[:, 0], true_c[:, 1], s=70, facecolors="none", edgecolors="C0", label="true camera")
    ax.scatter(aligned[:, 0], aligned[:, 1], s=14, color="C3", label="recovered by SfM")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("camera positions, seen from above", fontsize=9)
    ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[0, 3])
    ax.hist(rot, bins=12, color="C0")
    ax.axvline(2.0, color="C3", ls="--", lw=1.5)
    ax.text(2.05, ax.get_ylim()[1] * 0.85, "carving's budget\n2 deg", color="C3", fontsize=8)
    ax.set_xlabel("camera rotation error, degrees"); ax.set_title("median %.3f deg, worst %.3f" % (np.median(rot), max(rot)), fontsize=9)

    P = a["points_mm"]; Q = b["points_mm"]
    for j, (pts, name) in enumerate(((P, "carved from TRUE poses"), (Q, "carved from SfM poses"))):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(pts[:, 0], pts[:, 2], s=1.5, color="C0" if j == 0 else "C3")
        ax.set_aspect("equal"); ax.set_title("%s\n%d voxels" % (name, len(pts)), fontsize=9)
        ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[1, 2])
    only_a = np.array([p for p in A - B], float) * voxel
    only_b = np.array([p for p in B - A], float) * voxel
    both = np.array([p for p in A & B], float) * voxel
    if len(both):
        ax.scatter(both[:, 0], both[:, 2], s=1.5, color=(0.6, 0.6, 0.6), label="both")
    if len(only_a):
        ax.scatter(only_a[:, 0], only_a[:, 2], s=6, color="C0", label="true only")
    if len(only_b):
        ax.scatter(only_b[:, 0], only_b[:, 2], s=6, color="C3", label="SfM only")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("difference: agreement %.3f" % agree, fontsize=9)

    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0, 1.0, "what this gate settles", fontsize=12, fontweight="bold", va="top")
    ax.text(0, 0.9,
            "the board was only ever for POSE\n"
            "    carving, known poses ....... 0.782 solid IoU\n"
            "    photo pipeline .............  ~0.43\n\n"
            "SfM gives the same pose\n"
            "    rotation error ....... %.3f deg median\n"
            "    budget ...............  2 deg\n"
            "    a detected board ..... 0.016 deg\n"
            "    carve agreement ...... %.3f\n\n"
            "two hard requirements\n"
            "    quarter texture contrast -> 0 of 16\n"
            "        images registered, every part\n"
            "    8 views -> only 5 register\n\n"
            "untested: real photographs.\n"
            "blur, exposure drift, a real desk's\n"
            "texture. this gate is geometry only.\n\n"
            "the ask is now: put the part on a\n"
            "newspaper, take 16 photos walking\n"
            "round it. no printing, no target."
            % (np.median(rot), agree),
            fontsize=9, va="top", family="monospace")

    fig.suptitle("photo2fcstd - camera poses from the scene instead of a printed target (%s, %d views)"
                 % (part, views), fontsize=13)
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    plt.savefig(os.path.join(ROOT, out), dpi=70, bbox_inches="tight")
    print("wrote", out, "| agreement %.3f | rotation %.3f deg" % (agree, np.median(rot)))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["00005"]))
