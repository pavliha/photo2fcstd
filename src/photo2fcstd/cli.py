import argparse
import json
import os
import subprocess
import sys

from photo2fcstd import analysis, spec
from photo2fcstd.settings import FREECADCMD

BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.py")


def parse(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd", description="photos of a part in, a parametric FreeCAD document out")
    p.add_argument("photos", nargs="+")
    p.add_argument("--out", default="part.FCStd")
    p.add_argument("--stl")
    p.add_argument("--name")
    p.add_argument("--mode", choices=("stations", "profile", "plan", "revolve"))
    p.add_argument("--mm-per-px", type=float)
    p.add_argument("--length-mm", type=float)
    p.add_argument("--thickness-px", type=float)
    p.add_argument("--rim-px", type=float)
    p.add_argument("--min-step-px", type=float)
    p.add_argument("--grad", type=float)
    p.add_argument("--tail", type=float)
    p.add_argument("--segment", default="rmbg", choices=("rmbg", "auto", "dark"))
    p.add_argument("--spec-only", action="store_true")
    p.add_argument("--rectify", action="store_true", help="find the ChArUco target in each photo, flatten the perspective and take the scale from it")
    return p.parse_args(argv)


def freecad_build(spec_path, out):
    env = dict(os.environ, P2F_SPEC=spec_path, P2F_OUT=out)
    r = subprocess.run([FREECADCMD, BUILDER], capture_output=True, text=True, timeout=600, env=env)
    reports = [l for l in r.stdout.splitlines() if l.startswith("REPORT ")]
    if not reports:
        noise = ("FreeCAD", "(C) 2001")
        lines = [l for l in (r.stdout + r.stderr).splitlines() if not l.startswith(noise)]
        raise SystemExit("FreeCAD build failed:\n" + "\n".join(lines)[-1800:])
    return json.loads(reports[0][7:])


def main(argv):
    a = parse(argv)
    out = os.path.abspath(a.out)
    kw = {k: v for k, v in (("min_step_px", a.min_step_px), ("grad", a.grad), ("tail", a.tail)) if v is not None}
    views = [analysis.view(p, segment=a.segment, rectify=a.rectify, **kw) for p in a.photos[:3]]
    if a.rectify:
        found = sum(1 for v in views if v.get("mm_per_px"))
        print("rectified %d of %d photos from the target" % (found, len(views)))
    doc = spec.assemble(views, name=a.name or os.path.splitext(os.path.basename(out))[0], mode=a.mode,
                        mm_per_px=a.mm_per_px, length_mm=a.length_mm, thickness_px=a.thickness_px,
                        rim_px=a.rim_px, stl=os.path.abspath(a.stl) if a.stl else None)
    spec_path = os.path.splitext(out)[0] + ".spec.json"
    with open(spec_path, "w") as fh:
        json.dump(doc, fh)
    if a.spec_only:
        print("spec: %s" % spec_path)
        return doc
    report = freecad_build(spec_path, out)
    bad = {k: v for k, v in report["sketches"].items() if v["solve"] != 0 or v["redundant"] or v["conflicting"]}
    print("FCStd: %s  valid=%s solids=%d bbox=%s  sketches=%d%s"
          % (out, report["valid"], report["solids"], report["bbox"], len(report["sketches"]),
             "  PROBLEMS: %s" % bad if bad else "  all fully constrained"))
    return report


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
