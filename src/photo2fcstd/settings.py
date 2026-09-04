import os
import shutil

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FREECAD_NAMES = ("FreeCADCmd", "freecadcmd", "FreeCADCmd.exe")
FREECAD_GUESSES = (
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd",
    "/usr/bin/freecadcmd",
    "/usr/local/bin/FreeCADCmd",
    "/opt/conda/envs/fc/bin/freecadcmd",
    os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"),
)
DATA_GUESSES = (
    os.path.join(PACKAGE_ROOT, "data", "printcad", "PrintCAD"),
)


def bop_dir():
    return os.path.expanduser(os.environ.get("P2F_BOP", os.path.join(PACKAGE_ROOT, "data", "bop")))


def charuco_target():
    return os.path.join(PACKAGE_ROOT, "data", "charuco_target.png")


def find_freecad():
    fromenv = os.environ.get("FREECADCMD")
    if fromenv:
        return fromenv
    onpath = next((shutil.which(name) for name in FREECAD_NAMES if shutil.which(name)), None)
    if onpath:
        return onpath
    return next((p for p in FREECAD_GUESSES if os.path.exists(p)), FREECAD_GUESSES[0])


def data_dir():
    fromenv = os.environ.get("P2F_DATA")
    if fromenv:
        return os.path.expanduser(fromenv)
    return next((p for p in DATA_GUESSES if os.path.isdir(p)), DATA_GUESSES[0])


def cache_dir(kind):
    return os.path.join(os.path.expanduser("~"), ".cache", "photo2fcstd", kind)


FREECADCMD = find_freecad()
