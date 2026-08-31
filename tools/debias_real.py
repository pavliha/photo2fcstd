"""Does the height correction help on real parts, or only on flat-topped test blocks?"""
import json, os, sys
from multiprocessing import Pool

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2fcstd import capture_check as CK, carve as C, sketch_score as SS  # noqa: E402
from photo2fcstd.bench import truth_of  # noqa: E402

IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    try:
        w_mm, h_mm = CK.board_mm()
        mesh = trimesh.load(truth_of(part))
        mesh.apply_translation(-mesh.bounds[0])
        ext = mesh.extents.copy()
        if max(ext) > 70 or min(ext) < 1.0:
            return None
        mesh.apply_translation([w_mm / 2 - ext[0] / 2, h_mm / 2 - ext[1] / 2, 0.0])
        views = CK.board_views(16, radius=max(w_mm, h_mm) * 1.6, elevations=CK.DETECTABLE_ELEV)
        rows = CK.recover(views, mesh=mesh)
        found = [r for r in rows if r["got"]]
        if len(found) < 8:
            return None
        carved = C.carve([r["got"] for r in found], [r["mask"] for r in found], voxel_mm=0.5,
                         bounds=((w_mm / 2 - 40, w_mm / 2 + 40), (h_mm / 2 - 40, h_mm / 2 + 40),
                                 (0.0, float(ext[2]) * 2 + 10)))
        if carved is None or len(carved["points_mm"]) < 50:
            return None
        carved["min_elevation_deg"] = min(C.view_elevation_deg(r["got"]) for r in found)
        raw = float(np.ptp(carved["points_mm"][:, 2]))
        fixed = float(np.ptp(C.debias_height(carved)["points_mm"][:, 2]))
        top = carved["points_mm"]
        z = top[:, 2]
        cap = top[z >= z.max() - 2.0]
        mid = top[np.abs(z - (z.max() - 0.35 * np.ptp(z))) < 1.0]
        flat = (len(cap) / max(len(mid), 1)) if len(mid) else 0.0
        return {"part": part, "true": float(ext[2]), "raw": raw, "fixed": fixed, "flatness": float(flat)}
    except Exception:
        return None


def main(limit=60):
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])][:limit]
    with Pool(6) as pool:
        out = [r for r in pool.map(one, parts) if r]
    json.dump(out, open(os.path.join(ROOT, "data", "debias_real.json"), "w"))
    t = np.array([r["true"] for r in out])
    raw = np.array([r["raw"] for r in out]) - t
    fix = np.array([r["fixed"] for r in out]) - t
    flat = np.array([r["flatness"] for r in out])
    print("height error over %d real parts carved from rendered board photos\n" % len(out))
    print("  %-28s %10s %10s" % ("", "raw hull", "debiased"))
    print("  %-28s %+10.2f %+10.2f" % ("mean bias (mm)", raw.mean(), fix.mean()))
    print("  %-28s %10.2f %10.2f" % ("mean absolute error (mm)", np.abs(raw).mean(), np.abs(fix).mean()))
    print("  %-28s %10.2f %10.2f" % ("median absolute error (mm)", np.median(np.abs(raw)), np.median(np.abs(fix))))
    print("  %-28s %10d %10d" % ("parts within 1 mm", (np.abs(raw) < 1).sum(), (np.abs(fix) < 1).sum()))
    hi = flat >= np.median(flat)
    print("\n  split on how flat the hull's top is:")
    for name, sel in (("flat topped", hi), ("tapered", ~hi)):
        print("    %-16s n=%2d  raw %+.2f  debiased %+.2f mm"
              % (name, sel.sum(), raw[sel].mean(), fix[sel].mean()))
    print("\n  debias helps on %d parts, hurts on %d"
          % (int((np.abs(fix) < np.abs(raw)).sum()), int((np.abs(fix) > np.abs(raw)).sum())))


if __name__ == "__main__":
    main()
