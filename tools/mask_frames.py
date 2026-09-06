import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def without_skin(mask, image):
    import cv2
    from scipy import ndimage
    ycc = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2YCrCb)
    skin = (ycc[..., 1] > 133) & (ycc[..., 1] < 173) & (ycc[..., 2] > 77) & (ycc[..., 2] < 127)
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    m = mask & ~skin
    lab, k = ndimage.label(m)
    if k > 1:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, k + 1))) + 1)
    return ndimage.binary_fill_holes(m)


def main(frames_dir, masks_dir):
    from photo2fcstd.trace import segment_photo
    os.makedirs(masks_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        m = without_skin(segment_photo(f) > 0, Image.open(f).convert("RGB"))
        Image.fromarray((m > 0).astype(np.uint8) * 255).save(os.path.join(masks_dir, os.path.splitext(os.path.basename(f))[0] + ".png"))
        print(os.path.basename(f), round(float(m.mean()), 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
