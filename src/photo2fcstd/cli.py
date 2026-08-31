import argparse
import json
import os
import subprocess
import sys

from photo2fcstd import analysis, spec
from photo2fcstd.errors import BuildError, CaptureError
from photo2fcstd.settings import FREECADCMD

BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.py")


def parse(argv):
    p = argparse.ArgumentParser(
        prog="photo2fcstd",
        description="Photos of a part in, a parametric FreeCAD document out: constrained sketches "
                    "bound to a spreadsheet, one named dimension per feature.",
        epilog="Two or three photos work best: two elevations, or every photo of the same face for a plate. "
               "Shoot on the printed ChArUco target and add --rectify to get the scale and square-on geometry for free.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("photos", nargs="+", help="one to three photos of the same part")
    p.add_argument("--out", default="part.FCStd", help="where to write the FreeCAD document")
    p.add_argument("--stl", help="also export a mesh here, for checking or printing")
    p.add_argument("--name", help="name for the body and the document (default: the output filename)")
    p.add_argument("--mode", choices=("stations", "profile", "plan", "revolve"),
                   help="force how the part is modelled instead of deciding from the photos")
    p.add_argument("--rectify", action="store_true",
                   help="find the ChArUco target in each photo, flatten the perspective and take the scale from it")
    p.add_argument("--square-mm", type=float, metavar="MM",
                   help="size of one target square as you measured it - needed when the target is shown on a screen "
                        "or printed at the wrong scale (default: %.1f)" % 15.0)
    scale = p.add_argument_group("scale (pick one, or the sheet stays in pixels)")
    scale.add_argument("--mm-per-px", type=float, metavar="MM", help="millimetres per pixel, if you already know it")
    scale.add_argument("--length-mm", type=float, metavar="MM", help="the part's longest dimension, measured with a caliper")
    measured = p.add_argument_group("dimensions the photos cannot show")
    measured.add_argument("--thickness-px", type=float, metavar="PX",
                          help="plate thickness or extrusion depth, in pixels of the first photo")
    measured.add_argument("--rim-px", type=float, metavar="PX", help="rim height for a revolved part, in pixels")
    tuning = p.add_argument_group("tracing (defaults are fitted on the benchmark)")
    tuning.add_argument("--min-step-px", type=float, metavar="PX",
                        help="ignore steps in the silhouette shorter than this (default: 4%% of the part length)")
    tuning.add_argument("--grad", type=float, help="how sharply the width must change to start a new station")
    tuning.add_argument("--tail", type=float, help="fraction of each end to ignore, where the silhouette is noisiest")
    tuning.add_argument("--segment", default="rmbg", choices=("rmbg", "auto", "dark"),
                        help="how to separate the part from the background")
    p.add_argument("--spec-only", action="store_true", help="write the JSON spec and stop, without running FreeCAD")
    return p.parse_args(argv)


def freecad_build(spec_path, out):
    if not os.path.exists(FREECADCMD):
        raise BuildError("no FreeCAD at %s - install FreeCAD and set FREECADCMD to its FreeCADCmd binary" % FREECADCMD)
    env = dict(os.environ, P2F_SPEC=spec_path, P2F_OUT=out)
    r = subprocess.run([FREECADCMD, BUILDER], capture_output=True, text=True, timeout=600, env=env)
    reports = [l for l in r.stdout.splitlines() if l.startswith("REPORT ")]
    if not reports:
        noise = ("FreeCAD", "(C) 2001")
        lines = [l for l in (r.stdout + r.stderr).splitlines() if not l.startswith(noise)]
        raise BuildError("FreeCAD build failed:\n" + "\n".join(lines)[-1800:])
    return json.loads(reports[0][7:])


def readable(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise CaptureError("cannot find %s" % ", ".join(missing))
    return paths


def main(argv):
    a = parse(argv)
    if a.square_mm:
        os.environ["P2F_SQUARE_MM"] = str(a.square_mm)
        import importlib
        from photo2fcstd import capture, make_target, rectify
        for module in (make_target, rectify, capture):
            importlib.reload(module)
        print("target squares taken as %.2f mm" % a.square_mm)
    out = os.path.abspath(a.out)
    readable(a.photos)
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
    from photo2fcstd.errors import Photo2FCStdError
    try:
        main(sys.argv[1:])
    except Photo2FCStdError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    run()
