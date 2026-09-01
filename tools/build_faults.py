"""The four parts that build wrong, in detail, so the two faults can be told apart.

00011 and 00039 produce nothing and their sketch solver returns -2 and -5. 00061 and 00086 produce
a solid *with volume* that `isValid()` rejects - which matters more, because "valid" in a build
report means non-null with volume, so those two pass the repo's own check while being wrong.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FREECAD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
PARTS = ["00011", "00039", "00061", "00086"]


def main(parts=None):
    from photo2fcstd import analysis, bench, spec as spec_mod
    parts = parts or PARTS
    run = os.path.join(ROOT, "runs", "faults", "out")
    os.makedirs(run, exist_ok=True)
    rows = []
    for part in parts:
        views = [analysis.view(p) for p in bench.photos_of(part)[:3]]
        doc = spec_mod.assemble(views, name=part, log=lambda *a: None)
        sp = os.path.join(run, part + ".spec.json")
        json.dump(doc, open(sp, "w"))
        ol = doc.get("outline") or {}
        loops = ol.get("loops", [])
        kinds = {}
        for l in loops:
            if l["type"] == "circle":
                kinds["circle"] = kinds.get("circle", 0) + 1
            else:
                for e in l["elements"]:
                    kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        print("  %s mode=%-8s loops=%d %s depth=%s"
              % (part, doc.get("mode"), len(loops), kinds, doc.get("params", {}).get("depth")
                 if isinstance(doc.get("params"), dict) else "-"))
        rows.append("%s\t%s" % (sp, os.path.join(run, part + ".FCStd")))
    listing = os.path.join(ROOT, "runs", "faults", "list.txt")
    open(listing, "w").write("\n".join(rows))
    proc = subprocess.run([FREECAD, os.path.join(ROOT, "src", "photo2fcstd", "build.py")],
                          env=dict(os.environ, P2F_LIST=listing), capture_output=True, text=True, timeout=1800)
    print()
    for line in proc.stdout.splitlines():
        if line.startswith("BATCH "):
            _, name, blob = line.split(" ", 2)
            r = json.loads(blob)
            print("  %s valid=%s solids=%s volume=%s bbox=%s"
                  % (name, r.get("valid"), r.get("solids"), r.get("volume"), r.get("bbox")))
            for sk, d in r.get("sketches", {}).items():
                print("      %-12s solve=%s dof=%s geometry=%s constraints=%s redundant=%s conflicting=%s"
                      % (sk, d.get("solve"), d.get("dof"), d.get("geometry"), d.get("constraints"),
                         d.get("redundant"), d.get("conflicting")))
            if r.get("error"):
                print("      error: %s" % r["error"])
    err = [l for l in proc.stderr.splitlines() if "Error" in l or "error" in l or "Warning" in l]
    if err:
        print("\n  stderr highlights:")
        for l in err[:12]:
            print("   ", l[:150])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
