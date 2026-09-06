import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def spec_from_mask(mask, part):
    from scipy import ndimage
    from photo2fcstd import analysis, spec as spec_mod
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    lab, k = ndimage.label(m)
    if k > 1:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, k + 1))) + 1)
    m = ndimage.binary_fill_holes(m)
    view = analysis.view_from_mask(m)
    loops = spec_mod.traced_outline(view)
    if not loops:
        return None
    return {"name": part, "mode": "plan", "mm_per_px": 1.0, "unit": "px", "scale_note": "", "views": {},
            "outline": {"source": "faceview", "loops": loops, "depth_px": 10.0, "depth_note": "", "depth_trusted": False},
            "revolve": None, "stl": None, "measured": []}


def main(out_dir):
    from photo2fcstd import bench, recognise, sketch_score as SS
    rows = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "*_faceview.png"))):
        part = os.path.basename(fp).split("_")[0]
        photos = bench.photos_of(part)[:3]
        best_photo = 0.0
        for p in photos:
            try:
                s, _ = recognise.route([p], name=part, rec={"single_extrusion": True, "face_photo_index": 0})
                best_photo = max(best_photo, SS.score_one(s, IDEAL[part])["region_iou"])
            except Exception:
                pass
        r = {"part": part, "best_photo_iou": best_photo}
        for tag in ("faceview", "topview"):
            m = cv2.imread(os.path.join(out_dir, "%s_%s.png" % (part, tag)), 0) > 128
            try:
                spec = spec_from_mask(m, part)
                sc = SS.score_one(spec, IDEAL[part]) if spec else None
                r[tag + "_iou"] = sc["region_iou"] if sc else None
                r[tag + "_f1"] = (sc.get("primitive_f1") or {}).get("f1") if sc else None
            except Exception as e:
                r[tag + "_iou"] = None; r[tag + "_err"] = str(e)[:80]
        rect = []
        for rp in sorted(glob.glob(os.path.join(out_dir, "%s_rect_v*.png" % part))):
            m = cv2.imread(rp, 0) > 128
            try:
                spec = spec_from_mask(m, part); sc = SS.score_one(spec, IDEAL[part]) if spec else None
                rect.append(sc["region_iou"] if sc else 0.0)
            except Exception:
                rect.append(0.0)
        r["rect_views_iou"] = rect; r["rect_best_iou"] = max(rect) if rect else None
        rows.append(r)
        print("FACE %s best_photo %.3f | rectified views %s -> best %s | faceview %s | topview %s" % (
            part, best_photo, [round(x, 3) for x in rect], r["rect_best_iou"], r.get("faceview_iou"), r.get("topview_iou")))
    ok = [r for r in rows if r.get("faceview_iou") is not None]
    if ok:
        rb = [r for r in rows if r.get("rect_best_iou") is not None]
        print("SUMMARY n=%d  best_photo mean %.3f | rectified-best mean %.3f (wins %d/%d) | faceview mean %.3f | topview mean %.3f"
              % (len(ok), np.mean([r["best_photo_iou"] for r in ok]), np.mean([r["rect_best_iou"] for r in rb]) if rb else -1,
                 sum(r["rect_best_iou"] > r["best_photo_iou"] for r in rb), len(rb), np.mean([r["faceview_iou"] for r in ok]),
                 np.mean([r["topview_iou"] for r in ok if r.get("topview_iou") is not None])))
    json.dump(rows, open(os.path.join(out_dir, "face_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
