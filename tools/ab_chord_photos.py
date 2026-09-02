"""The ellipse primitive on real photographs, with a build check.

Set the flag for *both* arms on every job. A pool worker is reused, so patching the control arm
only leaves that worker patched for every later job and both arms measure the control - which is
what produced a null agreeing to four decimals, builds included.

Perfect-input arms said +0.0030 structure with no precision cost. That is measured on rasterised
faces; thresholds here are jointly tuned and several changes have improved a statistic and broken
models, so this regenerates specs for both arms on the same photographs and builds them.
"""
import json, os, subprocess, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
ARMS = {"before": False, "ellipse": True}
CURVED = ("arc", "circle", "ellipse", "bsplinecurve")


def one(args):
    part, arm = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, trace
    trace.ELLIPSE_ARCS = ARMS[arm]
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        v = SS.verdict(s)
        v["curve_mine"] = sum(c for k, c in s["counts_mine"].items() if k in CURVED)
        v["curve_ideal"] = sum(c for k, c in s["counts_ideal"].items() if k in CURVED)
        v["spec"] = doc
        return part, arm, v
    except Exception as e:
        return part, arm, {"error": str(e)[:60]}


def build(specs, tag):
    run = os.path.join(ROOT, "runs", "chord_" + tag, "out")
    os.makedirs(run, exist_ok=True)
    listing = os.path.join(ROOT, "runs", "chord_" + tag, "list.txt")
    rows = []
    for part, doc in specs.items():
        sp = os.path.join(run, part + ".spec.json")
        json.dump(doc, open(sp, "w"))
        rows.append("%s\t%s" % (sp, os.path.join(run, part + ".FCStd")))
    open(listing, "w").write("\n".join(rows))
    proc = subprocess.run([FREECAD, os.path.join(ROOT, "src", "photo2fcstd", "build.py")],
                          env=dict(os.environ, P2F_LIST=listing), capture_output=True, text=True, timeout=3600)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("BATCH "):
            _, name, blob = line.split(" ", 2)
            out[name] = json.loads(blob)
    return out


def main(limit=160):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    jobs = [(p, a) for a in ARMS for p in parts]
    with Pool(6) as pool:
        rows = [r for r in pool.map(one, jobs)]
    by = {}
    for part, arm, v in rows:
        by.setdefault(arm, {})[part] = v
    keen = [p for p in parts if all("region_iou" in by[a].get(p, {}) for a in ARMS)
            and by["before"][p]["discriminating"]]
    print("  n=%d discriminating parts, specs regenerated for both arms\n" % len(keen))
    print("  %-10s %9s %9s %10s %10s %9s" % ("arm", "IoU", "structure", "curves", "elements t", "exact"))
    for arm in ARMS:
        d = by[arm]
        print("  %-10s %9.3f %9.3f %10.2f %10.3f %8.0f%%"
              % (arm, np.mean([d[p]["region_iou"] for p in keen]), np.mean([d[p]["structure"] for p in keen]),
                 np.mean([d[p]["curve_mine"] for p in keen]), np.mean([d[p]["elements"] for p in keen]),
                 100 * np.mean([d[p]["exact"] for p in keen])))
    print("  %-10s %9s %9s %10.2f" % ("really", "-", "-", np.mean([by["before"][p]["curve_ideal"] for p in keen])))
    for key in ("region_iou", "structure"):
        d = np.array([by["ellipse"][p][key] - by["before"][p][key] for p in keen])
        m, lo, hi = stats.mean_ci(d)
        print("\n  %-12s ellipse vs before %+.4f [%+.4f, %+.4f]" % (key, m, lo, hi))
    print("\n  building both arms through FreeCAD...")
    for arm in ARMS:
        specs = {p: by[arm][p]["spec"] for p in keen}
        rep = build(specs, arm)
        valid = sum(1 for r in rep.values() if r.get("valid"))
        sk = [s for r in rep.values() for s in r.get("sketches", {}).values()]
        free = sum(1 for s in sk if s.get("dof"))
        unsolved = sum(1 for s in sk if s.get("solve") != 0)
        print("  %-10s built %d/%d, valid solid %d (%.0f%%), sketches unsolved %d, free DoF %d"
              % (arm, len(rep), len(specs), valid, 100 * valid / max(len(rep), 1), unsolved, free))
    for a in by:
        for v in by[a].values():
            v.pop("spec", None)
    json.dump(by, open(os.path.join(ROOT, "data", "ab_chord_photos.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
