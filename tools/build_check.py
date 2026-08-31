"""Build a sample of specs through FreeCAD and report null solids and loose sketches."""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import analysis, bench, spec as spec_mod  # noqa: E402
from photo2fcstd.settings import data_dir  # noqa: E402

FREECADCMD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
IDEAL = json.load(open(os.path.join(ROOT, "data", "printcad_ideal_sketches_all.json")))


def main(arm, n=45):
    out_dir = os.path.join(ROOT, "runs", "build_check_%s" % arm, "out")
    shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    scored = json.load(open(os.path.join(ROOT, "data", "ab_view_%s.json" % arm)))
    parts = [k for k, v in scored.items() if "iou" in v][:n]
    listing = os.path.join(out_dir, "list.tsv")
    made = 0
    with open(listing, "w") as fh:
        for part in parts:
            try:
                views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
                doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
                sp = os.path.join(out_dir, part + ".spec.json")
                json.dump(doc, open(sp, "w"))
                fh.write("%s\t%s\n" % (sp, os.path.join(out_dir, part + ".FCStd")))
                made += 1
            except Exception:
                continue
    log = os.path.join(out_dir, "build.log")
    with open(log, "w") as fh:
        subprocess.run([FREECADCMD, os.path.join(ROOT, "src", "photo2fcstd", "build.py")],
                       stdout=fh, stderr=subprocess.STDOUT,
                       env=dict(os.environ, P2F_LIST=listing), timeout=60 * made + 120)
    text = open(log, errors="ignore").read()
    valid = text.count("valid=True")
    invalid = text.count("valid=False")
    loose = text.lower().count("not fully constrained")
    print("%s: %d specs, built %d valid, %d null or invalid, %d not fully constrained"
          % (arm, made, valid, invalid, loose))
    return valid, invalid, loose


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shipped")
