import argparse
import sys

from photo2fcstd import bop


def main(argv):
    p = argparse.ArgumentParser(
        prog="photo2fcstd-bop",
        description="Carve BOP objects from their real photographs and known poses, and compare "
                    "the result against the CAD model in millimetres.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("objects", nargs="*", type=int, help="object ids (default: 1..30)")
    p.add_argument("--dataset", default="tless")
    p.add_argument("--split", default="train_primesense")
    p.add_argument("--views", type=int, default=24, help="how many of the scene's photos to use")
    p.add_argument("--voxel-mm", type=float, default=1.0)
    p.add_argument("--allow-misses", type=int, default=1, help="views a voxel may fall outside before it is carved away")
    p.add_argument("--depth", action="store_true", help="also carve away empty space seen by the depth maps, which recovers concavities")
    a = p.parse_args(argv)
    ids = a.objects or list(range(1, 31))
    rows = []
    for obj in ids:
        try:
            carved = bop.carve_object(obj, a.dataset, a.split, a.views, a.voxel_mm,
                                      allow_misses=a.allow_misses, use_depth=a.depth)
            row = bop.compare_to_truth(carved, obj)
        except Exception as exc:
            print("obj %02d  failed: %s" % (obj, str(exc)[:80]))
            continue
        rows.append(row)
        print("obj %02d  %d views  carved %s mm  truth %s mm  worst error %+.2f mm  volume x%.2f"
              % (obj, row["views"], row["extents_mm"], row["truth_mm"], row["worst_mm"], row["volume_ratio"]))
    if rows:
        import numpy as np
        worst = np.array([r["worst_mm"] for r in rows])
        print("\n%d objects: median worst-dimension error %.2f mm, mean %.2f mm, max %.2f mm"
              % (len(rows), np.median(worst), worst.mean(), worst.max()))
    return rows


def run():
    from photo2fcstd.errors import Photo2FCStdError
    try:
        main(sys.argv[1:])
    except Photo2FCStdError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    run()
