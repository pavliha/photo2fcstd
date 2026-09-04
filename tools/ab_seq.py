"""A/B for the breakpoint model: spans from seqnet, fitting geometric, on photographs with builds.

The flag is set for both arms on every job. Inference runs on CPU inside the pool workers.
"""
import json, os, subprocess, sys

os.environ["P2F_SEQ_DEVICE"] = "cpu"
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
CURVES = ("arc", "circle", "ellipse", "bsplinecurve")
ARMS = ("off", "on")


def measure(doc, part):
    from photo2fcstd import sketch_score as SS
    s = SS.score_one(doc, IDEAL[part])
    v = SS.verdict(s)
    m, i = s["counts_mine"], s["counts_ideal"]
    v["curve_mine"] = sum(c for k, c in m.items() if k in CURVES)
    v["curve_ideal"] = sum(c for k, c in i.items() if k in CURVES)
    f1 = s.get("primitive_f1") or {}
    v["f1"] = float(f1.get("f1", 0.0))
    return v


def perfect_one(args):
    part, arm = args
    from photo2fcstd import analysis, spec as spec_mod, trace
    from tracer_ceiling import rasterise
    trace.SEQNET = arm == "on"
    try:
        mask = rasterise(IDEAL[part])
        if mask is None or mask.sum() < 500:
            return part, arm, None
        view = analysis.view_from_mask(mask.astype(np.uint8))
        loops = spec_mod.traced_outline(view)
        if not loops:
            return part, arm, None
        return part, arm, measure({"outline": {"loops": loops}}, part)
    except Exception as e:
        return part, arm, {"error": str(e)[:60]}


def photo_one(args):
    part, arm = args
    from photo2fcstd import analysis, bench, spec as spec_mod, trace
    trace.SEQNET = arm == "on"
    try:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        v = measure(doc, part)
        v["spec"] = doc
        return part, arm, v
    except Exception as e:
        return part, arm, {"error": str(e)[:60]}


def build(specs, tag):
    run = os.path.join(ROOT, "runs", "seq_" + tag, "out")
    os.makedirs(run, exist_ok=True)
    listing = os.path.join(ROOT, "runs", "seq_" + tag, "list.txt")
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


def report(by, keen, stage):
    from photo2fcstd import stats
    print("\n%s: n=%d\n" % (stage, len(keen)))
    print("  %-6s %9s %9s %9s %8s %10s %8s" % ("arm", "IoU", "structure", "prim F1", "curves", "elements t", "exact"))
    for arm in ARMS:
        d = by[arm]
        print("  %-6s %9.3f %9.3f %9.3f %8.2f %10.3f %7.0f%%"
              % (arm, np.mean([d[p]["region_iou"] for p in keen]), np.mean([d[p]["structure"] for p in keen]),
                 np.mean([d[p]["f1"] for p in keen]), np.mean([d[p]["curve_mine"] for p in keen]),
                 np.mean([d[p]["elements"] for p in keen]), 100 * np.mean([d[p]["exact"] for p in keen])))
    print("  really %27s %8.2f" % ("", np.mean([by["off"][p]["curve_ideal"] for p in keen])))
    for key in ("region_iou", "structure", "f1", "curves"):
        d = np.array([by["on"][p][key] - by["off"][p][key] for p in keen], float)
        m, lo, hi = stats.mean_ci(d)
        print("  %-12s on vs off %+.4f [%+.4f, %+.4f]  changed %d" % (key, m, lo, hi, int((d != 0).sum())))


def main(limit=420, do_photos=1):
    from photo2fcstd import bench, sketch_score as SS
    parts = [p for p in sorted(IDEAL) if SS.trustworthy(IDEAL[p]) and IDEAL[p].get("loops")][:limit]
    jobs = [(p, a) for a in ARMS for p in parts]
    with Pool(6) as pool:
        rows = pool.map(perfect_one, jobs)
    by = {}
    for part, arm, v in rows:
        by.setdefault(arm, {})[part] = v
    keen = [p for p in parts if all(by[a].get(p) and "region_iou" in by[a][p] for a in ARMS)]
    complex_ = [p for p in keen if by["off"][p]["curve_ideal"] >= 4]
    report(by, keen, "PERFECT INPUT, all")
    report(by, complex_, "PERFECT INPUT, four or more real curves")
    if not do_photos:
        return
    photo_parts = bench.with_photos(parts)[0]
    jobs = [(p, a) for a in ARMS for p in photo_parts]
    with Pool(6) as pool:
        rows = pool.map(photo_one, jobs)
    pby = {}
    for part, arm, v in rows:
        pby.setdefault(arm, {})[part] = v
    keen = [p for p in photo_parts if all(pby[a].get(p) and "region_iou" in pby[a][p] for a in ARMS)
            and pby["off"][p]["discriminating"]]
    report(pby, keen, "PHOTOGRAPHS")
    print("\n  building both arms through FreeCAD...")
    for arm in ARMS:
        rep = build({p: pby[arm][p]["spec"] for p in keen}, arm)
        valid = sum(1 for r in rep.values() if r.get("valid"))
        sk = [s for r in rep.values() for s in r.get("sketches", {}).values()]
        print("  %-6s built %d/%d, valid solid %d (%.0f%%), sketches unsolved %d, free DoF %d"
              % (arm, len(rep), len(keen), valid, 100 * valid / max(len(rep), 1),
                 sum(1 for s in sk if s.get("solve") != 0), sum(1 for s in sk if s.get("dof"))))
    for a in pby:
        for v in pby[a].values():
            v.pop("spec", None)
    json.dump(pby, open(os.path.join(ROOT, "data", "ab_seq.json"), "w"))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
