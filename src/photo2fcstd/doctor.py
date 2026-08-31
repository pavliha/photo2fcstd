import os
import subprocess
import sys

from photo2fcstd.settings import FREECADCMD, cache_dir, data_dir

REQUIRED = ("numpy", "scipy", "cv2", "PIL", "trimesh")
OPTIONAL = {"torch": "segmentation (RMBG-2.0)", "transformers": "segmentation (RMBG-2.0)",
            "skimage": "meshing carved volumes", "matplotlib": "galleries and overlays",
            "sklearn": "the learned mode selector", "shapely": "sketch scoring"}


def freecad_version():
    if not os.path.exists(FREECADCMD):
        return None
    probe = "import FreeCAD; print('.'.join(FreeCAD.Version()[:3]))"
    try:
        out = subprocess.run([FREECADCMD, "-c", probe], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return "unusable: %s" % exc
    line = [l.strip() for l in out.stdout.splitlines() if l.strip() and l.strip()[0].isdigit()]
    return line[-1] if line else "unusable, no version reported"


def module_present(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def checks():
    out = [("python", True, sys.version.split()[0])]
    missing = [m for m in REQUIRED if not module_present(m)]
    out.append(("required packages", not missing, "all present" if not missing else "missing " + ", ".join(missing)))
    for name, why in OPTIONAL.items():
        out.append(("optional: " + name, module_present(name), why))
    version = freecad_version()
    out.append(("FreeCAD", bool(version) and "unusable" not in version,
                "%s at %s" % (version or "not found", FREECADCMD) if version else
                "not found - install FreeCAD, or set FREECADCMD to its FreeCADCmd binary"))
    masks = cache_dir("masks")
    out.append(("mask cache", True, "%d cached in %s" % (len(os.listdir(masks)) if os.path.isdir(masks) else 0, masks)))
    data = data_dir()
    photos = os.path.join(data, "captured_img")
    out.append(("benchmark dataset", os.path.isdir(photos),
                data if os.path.isdir(photos) else "not present (only needed for the benchmark) - set P2F_DATA"))
    return out


def main(argv):
    rows = checks()
    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print("%s %-*s  %s" % ("ok  " if ok else "MISS", width, name, detail))
    essential = [name for name, ok, _ in rows if not ok and (name == "FreeCAD" or name == "required packages")]
    if essential:
        print("\ncannot run: %s" % ", ".join(essential))
        return 1
    print("\nready")
    return 0


def run():
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
