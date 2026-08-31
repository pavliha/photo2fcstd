import argparse
import json
import os
import sys

from photo2fcstd import carve, cli


def segment(path):
    from photo2fcstd.trace import segment_photo
    return segment_photo(path)


def main(argv):
    p = argparse.ArgumentParser(prog="photo2fcstd-carve",
                                description="Many photos of a part on the ChArUco target in, a model in real millimetres out: "
                                            "the board gives every photo a pose and a scale, so the height is measured rather than guessed.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("photos", nargs="+", help="ten or more photos of the part sitting on the ChArUco target, including low grazing angles")
    p.add_argument("--out", default="part.FCStd", help="where to write the FreeCAD document")
    p.add_argument("--stl", help="also export a mesh here")
    p.add_argument("--voxel-mm", type=float, default=carve.VOXEL_MM, help="carving resolution in millimetres")
    p.add_argument("--name", help="name for the body and the document")
    a = p.parse_args(argv)
    out = os.path.abspath(a.out)
    carved = carve.from_photos(a.photos, segment, a.voxel_mm)
    print("carved %d voxels from %d views, %.2f mm voxels, extents %s mm"
          % (len(carved["points_mm"]), carved["views"], carved["voxel_mm"],
             [round(float(x), 2) for x in carved["extents_mm"]]))
    doc = carve.spec_from_carve(carved, a.name or os.path.splitext(os.path.basename(out))[0],
                                os.path.abspath(a.stl) if a.stl else None)
    spec_path = os.path.splitext(out)[0] + ".spec.json"
    with open(spec_path, "w") as fh:
        json.dump(doc, fh)
    report = cli.freecad_build(spec_path, out)
    print("FCStd: %s  valid=%s solids=%d bbox=%s mm" % (out, report["valid"], report["solids"], report["bbox"]))
    return report


def run():
    from photo2fcstd.errors import Photo2FCStdError
    try:
        main(sys.argv[1:])
    except Photo2FCStdError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    run()
