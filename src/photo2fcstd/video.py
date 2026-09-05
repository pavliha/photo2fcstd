import os

import cv2
import numpy as np


def select_frames(video, out_dir, target=40, min_shift=0.06, sharp_quantile=0.5):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [f for ok, f in (cap.read() for _ in range(n)) if ok]
    cap.release()
    if not frames:
        raise ValueError("no frames in %s" % video)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in gray])
    floor = np.quantile(sharp, sharp_quantile)
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    w = frames[0].shape[1]
    kept, last = [], None
    for i, g in enumerate(gray):
        if sharp[i] < floor:
            continue
        kp, des = orb.detectAndCompute(g, None)
        if des is None:
            continue
        if last is not None:
            m = bf.match(last[1], des)
            if len(m) >= 12:
                shift = np.median([np.hypot(*(np.subtract(kp[x.trainIdx].pt, last[0][x.queryIdx].pt))) for x in m])
                if shift < min_shift * w:
                    continue
        kept.append(i)
        last = (kp, des)
    if len(kept) > target:
        kept = [kept[j] for j in np.linspace(0, len(kept) - 1, target).round().astype(int)]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, i in enumerate(kept):
        p = os.path.join(out_dir, "f%03d.jpg" % k)
        cv2.imwrite(p, frames[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    return {"total": n, "kept": len(paths), "sharp_floor": float(floor), "paths": paths}


def main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser(prog="photo2fcstd.video", description="select carve-grade frames from a video")
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--min-shift", type=float, default=0.06)
    a = ap.parse_args(argv)
    r = select_frames(a.video, a.out, a.target, a.min_shift)
    print(json.dumps({k: v for k, v in r.items() if k != "paths"}))


if __name__ == "__main__":
    main()
