"""Which structural term is the drawing losing, and does any part still draw nothing?

Two questions in one pass. `sketch_score.verdict` splits the drawing into loops, curves and
elements, so the answer to "where do we work next" is whichever term is furthest from 1. And
CLAUDE.md still describes a straight choice where `stations` leaves 46% of parts blank, which was
written before the mode allowed-list fix - this checks whether that choice still binds.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def one(part):
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        v = SS.verdict(s)
        v.update(mode=doc["mode"], drew=bool(doc.get("outline") or doc.get("revolve")))
        return part, v
    except Exception as e:
        return part, {"error": str(e)[:70]}


def main(limit=200):
    from photo2fcstd import bench, sketch_score as SS
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    with Pool(6) as pool:
        rows = dict(pool.map(one, parts))
    json.dump(rows, open(os.path.join(ROOT, "data", "where_wrong.json"), "w"))
    ok = {p: v for p, v in rows.items() if "region_iou" in v}
    keen = {p: v for p, v in ok.items() if v["discriminating"]}
    print("  %d parts, %d scored, %d discriminating\n" % (len(parts), len(ok), len(keen)))

    drew = sum(v["drew"] for v in ok.values())
    print("  drew a sketch: %d of %d (%.0f%%)" % (drew, len(ok), 100 * drew / max(len(ok), 1)))
    modes = {}
    for v in ok.values():
        modes.setdefault(v["mode"], []).append(v)
    print("  modes: %s\n" % ", ".join("%s %d" % (k, len(v)) for k, v in sorted(modes.items(), key=lambda x: -len(x[1]))))

    V = list(keen.values())
    print("  %-12s %8s" % ("term", "mean"))
    for term in ("region_iou", "structure", "loops", "curves", "elements"):
        print("  %-12s %8.3f" % (term, np.mean([v[term] for v in V])))
    print("  %-12s %7.0f%%" % ("exact", 100 * np.mean([v["exact"] for v in V])))
    print("\n  worst term is where the next change should aim")
    print("\n  %-10s %6s %8s %8s %8s %8s" % ("mode", "n", "IoU", "loops", "curves", "elements"))
    for k, v in sorted(modes.items(), key=lambda x: -len(x[1])):
        d = [x for x in v if x["discriminating"]]
        if not d:
            continue
        print("  %-10s %6d %8.3f %8.3f %8.3f %8.3f"
              % (k, len(d), np.mean([x["region_iou"] for x in d]), np.mean([x["loops"] for x in d]),
                 np.mean([x["curves"] for x in d]), np.mean([x["elements"] for x in d])))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
