"""Does the pipeline actually deliver its stated product, end to end?

Every measurement today was spec-only, which is fast and says nothing about whether FreeCAD can
open the result. This builds a sample the whole way and reports what the objective asks for: a
document that opens, a solid with volume, and a sketch a person could edit - which means solved,
and with its degrees of freedom reported rather than assumed.
"""
import json, os, subprocess, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))


def spec_of(part):
    from photo2fcstd import analysis, bench, spec as spec_mod
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        return part, spec_mod.assemble(views, name=part, log=lambda *a: None)
    except Exception as e:
        return part, {"error": str(e)[:70]}


def main(limit=60, run="runs/goal"):
    from photo2fcstd import bench, sketch_score as SS
    out_dir = os.path.join(ROOT, run, "out")
    os.makedirs(out_dir, exist_ok=True)
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    with Pool(6) as pool:
        specs = dict(pool.map(spec_of, parts))
    listing = os.path.join(ROOT, run, "list.txt")
    rows = []
    for part, doc in specs.items():
        if "error" in doc:
            continue
        sp = os.path.join(out_dir, part + ".spec.json")
        json.dump(doc, open(sp, "w"))
        rows.append("%s\t%s" % (sp, os.path.join(out_dir, part + ".FCStd")))
    open(listing, "w").write("\n".join(rows))
    print("  building %d specs through FreeCAD..." % len(rows))
    env = dict(os.environ, P2F_LIST=listing)
    proc = subprocess.run([FREECAD, os.path.join(ROOT, "src", "photo2fcstd", "build.py")],
                          env=env, capture_output=True, text=True, timeout=3600)
    reports = {}
    for line in proc.stdout.splitlines():
        if line.startswith("BATCH "):
            _, name, blob = line.split(" ", 2)
            reports[name] = json.loads(blob)
    if not reports:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:]); return
    valid = [r for r in reports.values() if r.get("valid")]
    sk = [s for r in reports.values() for s in r.get("sketches", {}).values()]
    solved = [s for s in sk if s.get("solve") == 0]
    free = [s for s in sk if s.get("dof")]
    print("\n  %-38s %d of %d (%.0f%%)" % ("documents that built", len(reports), len(rows), 100*len(reports)/max(len(rows),1)))
    print("  %-38s %d (%.0f%%)" % ("with a real solid (volume > 0)", len(valid), 100*len(valid)/max(len(reports),1)))
    print("  %-38s %d of %d (%.0f%%)" % ("sketches that solve", len(solved), len(sk), 100*len(solved)/max(len(sk),1)))
    print("  %-38s %d (%.0f%%)" % ("sketches with free DoF left", len(free), 100*len(free)/max(len(sk),1)))
    if free:
        print("  %-38s %.1f median" % ("  degrees of freedom when free", float(np.median([s["dof"] for s in free]))))
    scored = []
    for part, doc in specs.items():
        if "error" in doc or part not in reports:
            continue
        v = SS.verdict(SS.score_one(doc, IDEAL[part]))
        if v["discriminating"]:
            scored.append(v)
    if scored:
        print("\n  on the %d discriminating parts of this sample:" % len(scored))
        for k in ("region_iou", "structure", "loops", "curves", "elements"):
            print("    %-14s %.3f" % (k, np.mean([v[k] for v in scored])))
        print("    %-14s %.0f%%" % ("exact", 100*np.mean([v["exact"] for v in scored])))
    json.dump(reports, open(os.path.join(ROOT, "data", "goal_check.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]])
