"""MEASURED AND REVERTED. An escape on the circle gate, on real photographs, with a build check.

Two forms of the same hypothesis, and they disagree. An absolute-pixel escape wins on aggregate
(+0.0033 primitive F1 [+0.0001, +0.0072] at 5 px over 559 parts, builds identical) but admits small
rectangles as circles - a 40x20 rectangle fits an ellipse at rms 1.97 px, so any absolute threshold
takes it, and `test_every_loop_shares_the_part_frame` goes red. Loosening the *relative* threshold
instead keeps rectangles out at every setting and loses outright: F1 -0.0187 at 0.05, -0.0162 at
0.06, -0.0239 at 0.07, every CI clear of zero on the wrong side, because a 400 px loop at rms/a
0.05 is 20 px from circular and flattening it destroys the part.

So the absolute rule works only by being self-limiting to loops under 125 px, where PrintCAD's
holes happen to be round. That is a dataset prior, not a geometric fact. Reverted.

Run it with P2F_AB_ARMS to re-measure either form.

Set the flag for *both* arms on every job. A pool worker is reused, so patching only the control
arm leaves that worker patched for every later job and both arms measure the control.
"""
import json, os, subprocess, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
ARMS = {a: float(a) for a in os.environ.get("P2F_AB_ARMS", "0.04,0.06").split(",")}
CONTROL = sorted(ARMS)[0]


def one(args):
    part, arm = args
    from photo2fcstd import analysis, bench, sketch_score as SS, spec as spec_mod, trace
    if not hasattr(trace, "CIRCLE_RMS_FRAC"):
        raise SystemExit("re-apply the CIRCLE_RMS_FRAC knob to trace.ellipse_ok to run this")
    trace.CIRCLE_RMS_FRAC = ARMS[arm]
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        s = SS.score_one(doc, IDEAL[part])
        v = SS.verdict(s)
        f1 = s.get("primitive_f1") or {}
        v["f1"] = float(f1.get("f1", 0.0))
        v["drawn"] = int(f1.get("drawn", 0))
        v["wanted"] = int(f1.get("wanted", 0))
        v["circles"] = int(s["counts_mine"].get("circle", 0))
        v["circles_ideal"] = int(s["counts_ideal"].get("circle", 0))
        v["spec"] = doc
        return part, arm, v
    except Exception as e:
        return part, arm, {"error": str(e)[:60]}


def build(specs, tag):
    run = os.path.join(ROOT, "runs", "holegate_" + tag, "out")
    os.makedirs(run, exist_ok=True)
    listing = os.path.join(ROOT, "runs", "holegate_" + tag, "list.txt")
    rows = []
    for part, doc in specs.items():
        sp = os.path.join(run, part + ".spec.json")
        json.dump(doc, open(sp, "w"))
        rows.append("%s\t%s" % (sp, os.path.join(run, part + ".FCStd")))
    open(listing, "w").write("\n".join(rows))
    proc = subprocess.run([FREECAD, os.path.join(ROOT, "src", "photo2fcstd", "build.py")],
                          env=dict(os.environ, P2F_LIST=listing), capture_output=True, text=True, timeout=7200)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("BATCH "):
            _, name, blob = line.split(" ", 2)
            out[name] = json.loads(blob)
    return out


def main(limit=200, do_build=1):
    from photo2fcstd import bench, sketch_score as SS, stats
    parts = bench.with_photos([p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p])])[0][:limit]
    jobs = [(p, a) for a in ARMS for p in parts]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    by = {}
    for part, arm, v in rows:
        by.setdefault(arm, {})[part] = v
    keen = [p for p in parts if all("region_iou" in by[a].get(p, {}) for a in ARMS)
            and by[CONTROL][p]["discriminating"]]
    holed = [p for p in keen if by[CONTROL][p]["circles_ideal"] > 1]
    print("  n=%d discriminating parts, specs regenerated for both arms" % len(keen))
    print("  of which %d have more than one circle in the ideal\n" % len(holed))
    print("  %-10s %9s %9s %9s %10s %9s %8s" % ("arm", "IoU", "structure", "prim F1", "circles", "elements t", "exact"))
    for arm in ARMS:
        d = by[arm]
        print("  %-10s %9.3f %9.3f %9.3f %10.2f %10.3f %7.0f%%"
              % (arm, np.mean([d[p]["region_iou"] for p in keen]), np.mean([d[p]["structure"] for p in keen]),
                 np.mean([d[p]["f1"] for p in keen]), np.mean([d[p]["circles"] for p in keen]),
                 np.mean([d[p]["elements"] for p in keen]), 100 * np.mean([d[p]["exact"] for p in keen])))
    print("  %-10s %9s %9s %9s %10.2f" % ("really", "-", "-", "-",
                                          np.mean([by[CONTROL][p]["circles_ideal"] for p in keen])))
    for label, sel in (("all", keen), ("multi-circle", holed)):
        print()
        for ARM in [a for a in ARMS if a != CONTROL]:
            for key in ("region_iou", "structure", "f1"):
                d = np.array([by[ARM][p][key] - by[CONTROL][p][key] for p in sel], float)
                m, lo, hi = stats.mean_ci(d)
                fired = [p for p in sel if d[sel.index(p)] != 0] if False else None
                changed = int((d != 0).sum())
                print("  %-12s %-13s %-5s vs shipped %+.4f [%+.4f, %+.4f]  (n=%d, changed %d)"
                      % (key, label, ARM, m, lo, hi, len(sel), changed))
                if changed:
                    dc = d[d != 0]
                    m2, lo2, hi2 = stats.mean_ci(dc)
                    print("  %-12s %-13s %-5s on the %d that changed  %+.4f [%+.4f, %+.4f]"
                          % ("", "", "", changed, m2, lo2, hi2))
    if not do_build:
        return
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
    json.dump(by, open(os.path.join(ROOT, "data", "ab_hole_gate.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
