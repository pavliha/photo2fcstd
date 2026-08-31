"""The board's 15 degree detection floor overshoots height by a predictable amount. Test removing it."""
import os, sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import capture_check as CK, carve as C  # noqa: E402

SHAPES = [
    ("box 26x16x7", lambda: trimesh.creation.box((26.0, 16.0, 7.0))),
    ("box 26x16x3", lambda: trimesh.creation.box((26.0, 16.0, 3.0))),
    ("box 40x10x12", lambda: trimesh.creation.box((40.0, 10.0, 12.0))),
    ("box 18x18x18", lambda: trimesh.creation.box((18.0, 18.0, 18.0))),
    ("cylinder r9 h6", lambda: trimesh.creation.cylinder(radius=9.0, height=6.0)),
    ("cylinder r6 h20", lambda: trimesh.creation.cylinder(radius=6.0, height=20.0)),
    ("L-bracket", lambda: trimesh.util.concatenate([
        trimesh.creation.box((30.0, 12.0, 5.0)),
        trimesh.creation.box((5.0, 12.0, 16.0)).apply_translation([12.5, 0, 5.5])])),
]


def carve_shape(make, elevations):
    w_mm, h_mm = CK.board_mm()
    mesh = make()
    mesh.apply_translation(-mesh.bounds[0])
    true = mesh.extents.copy()
    mesh.apply_translation([w_mm / 2 - true[0] / 2, h_mm / 2 - true[1] / 2, 0.0])
    views = CK.board_views(16, radius=max(w_mm, h_mm) * 1.6, elevations=elevations)
    rows = CK.recover(views, mesh=mesh)
    found = [r for r in rows if r["got"]]
    carved = C.carve([r["got"] for r in found], [r["mask"] for r in found], voxel_mm=0.4,
                     bounds=((w_mm / 2 - 30, w_mm / 2 + 30), (h_mm / 2 - 25, h_mm / 2 + 25),
                             (0.0, float(true[2]) * 2 + 12)))
    carved["min_elevation_deg"] = min(C.view_elevation_deg(v) for v in views)
    return carved, true


def main():
    el = CK.DETECTABLE_ELEV
    print("board views at %s degrees, correction = width/2 * tan(%.0f)\n"
          % (str(el), el[0]))
    print("  %-16s %7s %8s %9s %9s %8s" % ("shape", "true h", "carved", "widest", "shipped", "error"))
    raw_err, cor_err, tip_err = [], [], []
    for name, make in SHAPES:
        carved, true = carve_shape(make, el)
        pts = carved["points_mm"]
        h = float(np.ptp(pts[:, 2]))
        overall = float(min(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        top = pts[pts[:, 2] >= pts[:, 2].max() - max(0.2 * h, 1.0)]
        near = float(min(np.ptp(top[:, 0]), np.ptp(top[:, 1]))) if len(top) > 4 else overall
        k = np.tan(np.radians(el[0])) / 2
        wide = max(h - overall * k, 0.4)
        tip = max(h - near * k, 0.4)
        shipped = float(np.ptp(C.debias_height(carved)["points_mm"][:, 2]))
        raw_err.append(h - true[2])
        cor_err.append(wide - true[2])
        tip_err.append(shipped - true[2])
        print("  %-16s %7.1f %8.1f %9.1f %9.1f %8.2f" % (name, true[2], h, wide, shipped, shipped - true[2]))
    print("\n  %-30s %s" % ("mean |error|", "carved %.2f  widest %.2f  shipped %.2f mm"
          % (np.mean(np.abs(raw_err)), np.mean(np.abs(cor_err)), np.mean(np.abs(tip_err)))))
    print("  %-30s %s" % ("bias", "carved %+.2f  widest %+.2f  shipped %+.2f mm"
          % (np.mean(raw_err), np.mean(cor_err), np.mean(tip_err))))


if __name__ == "__main__":
    main()
