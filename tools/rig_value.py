"""What the sixteen-photo rig would buy, measured before anyone shoots: carve-to-sketch on PrintCAD.

Renders sixteen true-pose views of each part's own truth mesh on a textured plane (the renderer the
SfM gate validated; SfM-vs-true-pose carve agreement is 0.987-0.997, so true poses stand in), carves,
slices the mid-depth cross-section, traces it with the production tracer, and scores the drawing
against the ideal sketch - paired with the photo pipeline's score on the identical parts. This is
the noise-floor thesis under test: decisions a single matted contour cannot make should become
decidable on a sixteen-view consensus section.
"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
VOXEL_MM = float(os.environ.get("P2F_RIG_VOXEL", 0.25))
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")


def section_mask(carved, axis, canvas=900):
    from scipy import ndimage
    from skimage import measure
    pts = carved["points_mm"]
    vox = carved["voxel_mm"]
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    mid = (lo + hi) / 2
    band = pts[np.abs(pts[:, axis] - mid) <= vox * 0.6]
    if len(band) < 20:
        return None
    keep = [i for i in range(3) if i != axis]
    ij = np.round(band[:, keep] / vox).astype(int)
    ij -= ij.min(axis=0)
    grid = np.zeros(ij.max(axis=0) + 3, np.float32)
    grid[ij[:, 0] + 1, ij[:, 1] + 1] = 1.0
    grid = ndimage.gaussian_filter(grid, 0.8)
    contours = [c for c in measure.find_contours(grid, 0.5) if len(c) >= 8]
    if not contours:
        return None
    span = max(grid.shape)
    scale = (canvas * 0.8) / span
    img = np.zeros((canvas, canvas), np.uint8)
    contours.sort(key=lambda c: -cv2.contourArea(np.round(c * scale).astype(np.int32)))
    for rank, c in enumerate(contours):
        q = np.round(c * scale + canvas * 0.1).astype(np.int32)[:, ::-1]
        cv2.fillPoly(img, [q], 0 if rank else 1)
    return img


def one(part):
    import trimesh
    import sfm_check as SC
    from photo2fcstd import analysis, bench, carve as C, sketch_score as SS, spec as spec_mod
    try:
        mesh = trimesh.load(bench.truth_of(part))
        thin = int(np.argmin(mesh.extents))
        if thin != 2:
            axes = [0, 1, 2]
            axes.remove(thin)
            R = np.eye(4)
            R[:3, :3] = np.eye(3)[[axes[0], axes[1], thin]].T
            mesh.apply_transform(R)
        mesh.apply_translation(-np.array([mesh.bounds[:, 0].mean(), mesh.bounds[:, 1].mean(),
                                          mesh.bounds[0][2]]))
        if mesh.extents[2] < 4 * VOXEL_MM:
            return part, {"skip": "thinner than four voxels"}
        radius = float(np.max(mesh.extents)) * 1.9
        views = SC.poses(16, radius)
        rng = np.random.default_rng(0)
        texture = cv2.GaussianBlur(rng.integers(30, 226, (600, 600, 3), dtype=np.uint8), (0, 0), 1.4)
        masks = [SC.render(mesh, v, texture, float(np.max(mesh.extents)) * 1.6, rng)[1] for v in views]
        bounds = [(float(mesh.bounds[0][k]) - 2, float(mesh.bounds[1][k]) + 2) for k in range(2)]             + [(0.0, float(mesh.bounds[1][2]) + 2)]
        carved = C.carve(views, masks, voxel_mm=VOXEL_MM, allow_misses=1,
                         bounds=bounds, max_height_mm=bounds[2][1])
        if carved is None:
            return part, {"skip": "carved nothing"}
        axis = 2
        m = section_mask(carved, axis)
        if m is None:
            return part, {"skip": "empty section"}
        view = analysis.view_from_mask(m)
        loops = spec_mod.traced_outline(view)
        if not loops:
            return part, {"skip": "no trace"}
        s = SS.score_one({"outline": {"loops": loops}}, IDEAL[part])
        f1 = s.get("primitive_f1") or {}
        return part, {"f1": float(f1.get("f1", 0.0)),
                      "curves": sum(c for k, c in s["counts_mine"].items() if k in CURVES),
                      "curves_ideal": sum(c for k, c in s["counts_ideal"].items() if k in CURVES),
                      "iou": s["region_iou"], "exact": s["counts_mine"] == s["counts_ideal"]}
    except Exception as e:
        return part, {"skip": str(e)[:60]}


def main(limit=80):
    from photo2fcstd import bench, sketch_score as SS, stats
    photo = bench.sketch_scores(os.path.join(ROOT, "runs", "chain_shipped"))
    parts = [p for p, r in sorted(photo.items())
             if r.get("trustworthy") and isinstance(r.get("primitive_f1"), dict)
             and r["primitive_f1"].get("wanted")][:limit]
    with Pool(6) as pool:
        rows = dict(pool.map(one, parts))
    ok = {p: v for p, v in rows.items() if "f1" in v}
    skips = [v.get("skip") for v in rows.values() if "skip" in v]
    print("parts: %d fed, %d carved and traced, %d skipped (%s)"
          % (len(parts), len(ok), len(skips),
             ", ".join("%s x%d" % (s, skips.count(s)) for s in sorted(set(skips)))))
    both = [(p, ok[p], photo[p]) for p in ok]
    import numpy as np
    pf = np.array([float(ph["primitive_f1"]["f1"]) for _, _, ph in both])
    cf = np.array([v["f1"] for _, v, _ in both])
    print("\n  %-28s %8s %8s" % ("same %d parts" % len(both), "photos", "rig"))
    print("  %-28s %8.3f %8.3f" % ("primitive F1", pf.mean(), cf.mean()))
    print("  %-28s %8.2f %8.2f" % ("curved primitives drawn",
          np.mean([ph.get("curve_mine", 0) or 0 for _, _, ph in both]) if any("curve_mine" in ph for _, _, ph in both) else float("nan"),
          np.mean([v["curves"] for _, v, _ in both])))
    print("  %-28s %8s %8.2f" % ("really there", "", np.mean([v["curves_ideal"] for _, v, _ in both])))
    print("  %-28s %7.0f%% %7.0f%%" % ("exact primitives",
          100 * np.mean([ph["primitive_f1"]["f1"] == 1.0 for _, _, ph in both]),
          100 * np.mean([v["exact"] for _, v, _ in both])))
    m, lo, hi = stats.mean_ci(cf - pf)
    print("\n  rig minus photos, primitive F1, paired: %+.4f [%+.4f, %+.4f]" % (m, lo, hi))
    json.dump({p: v for p, v, _ in [(p, v, None) for p, v in ok.items()]},
              open(os.path.join(ROOT, "data", "rig_value.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
