import os


def data_dir():
    return os.environ.get("P2F_DATA", os.path.expanduser("~/3DPrint/tools/data/printcad/PrintCAD"))


def cache_dir(kind):
    return os.path.join(os.path.expanduser("~"), ".cache", "photo2fcstd", kind)


FREECADCMD = os.environ.get("FREECADCMD", os.path.expanduser("~/Code/FreeCAD/build/release/bin/FreeCADCmd"))
