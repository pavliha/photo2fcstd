import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main(parts, out):
    from photo2fcstd import bench, multiview, recognise, sketch_score as SS, trace
    from photo2fcstd.trace import segment_photo
    RECS = json.load(open(os.path.join(ROOT, "runs", "bench_recs.json")))
    IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
    os.makedirs(out, exist_ok=True)
    for part in parts:
        photos = bench.photos_of(part)[:3]
        rec = {**RECS[part], "single_extrusion": True}
        spec = recognise.route(photos, name=part, rec=rec)[0]
        trace.RECOVER_DARK = False
        masks = [segment_photo(p) for p in photos]
        src = spec["outline"].get("source"); ref = photos.index(src) if src in photos else 0
        res = multiview.fit(masks, [multiview.focal_35(p) for p in photos], ref=ref)
        rect, _ = multiview.rectified_mask(res, masks[ref])
        spec2, info = multiview.maybe_rectify(photos, spec, name=part)
        cand = spec2 if info.get("applied") else info.get("candidate")
        np.savez_compressed(os.path.join(out, part + ".npz"), ref=ref, photo=photos[ref].encode(), P=res["P"], ious=res["ious"], tilts=res["tilts"],
                            rect=rect, masks_small=np.array([multiview._prep(m, 27.0)[0] for m in masks]),
                            base=np.array([r for r in SS.spec_rings(spec)], dtype=object), mv=np.array([r for r in SS.spec_rings(cand)], dtype=object) if cand else np.array([], dtype=object),
                            ideal=np.array(SS.ideal_rings(IDEAL[part]), dtype=object), allow_pickle=True)
        print(part, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
