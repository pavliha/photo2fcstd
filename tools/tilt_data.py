"""Is camera tilt in the photo's pixels, or only in its silhouette?

The capture term is entirely viewpoint tilt - eight degrees off the face normal costs 0.11 of
sketch IoU - and the pipeline currently has no way to know the tilt of a photo without a ChArUco
board in it. Segmentation throws away shading and specular highlights, which is where a surface
normal actually lives, so the question is whether the RGB carries tilt the mask does not.

T-LESS answers it with real photographs and exact poses. Both arms get the identical crop box
computed from the dataset mask, so the only difference is what the crop contains: the photograph,
or the binary silhouette of the same object in the same place. Held out by object, because
predicting the tilt of an object the model has memorised is not the problem we have.
"""
import json, os, sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from photo2fcstd import bop, embed  # noqa: E402

MARGIN = 0.15


def crop_box(mask, w, h):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = MARGIN * max(x1 - x0, y1 - y0)
    return (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
            min(w, int(x1 + pad)), min(h, int(y1 + pad)))


def shape_features(mask, box):
    sub = mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.0] * 9
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    (_, _), (bw, bh), _ = cv2.minAreaRect(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    hu = cv2.HuMoments(cv2.moments(c)).ravel()[:4]
    return [area / max(sub.size, 1), min(bw, bh) / max(bw, bh, 1e-6),
            area / max(bw * bh, 1e-6), area / max(hull, 1e-6),
            cv2.arcLength(c, True) ** 2 / max(area, 1e-6)] + [
            float(-np.sign(v) * np.log10(abs(v) + 1e-30)) for v in hu]


def rows_for(obj_id, dataset, split, stride):
    path = bop.scene_dir(dataset, split, obj_id)
    views = bop.views_of(path, stride=stride)
    dirs = bop.camera_directions(path)
    masks = bop.masks_for(path, views)
    rgb, obj, sil, feats, tilt, keys = [], [], [], [], [], []
    for v, m in zip(views, masks):
        if m.sum() < 200:
            continue
        image = Image.open(v["image"]).convert("RGB")
        box = crop_box(m, image.width, image.height)
        rgb.append(image.crop(box).resize((embed.SIZE, embed.SIZE)))
        cut = np.asarray(image)[box[1]:box[3], box[0]:box[2]] * m[box[1]:box[3], box[0]:box[2], None]
        obj.append(Image.fromarray(cut.astype(np.uint8)).resize((embed.SIZE, embed.SIZE)))
        s = (m[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255)
        sil.append(Image.fromarray(np.dstack([s] * 3)).resize((embed.SIZE, embed.SIZE)))
        feats.append(shape_features(m, box))
        d = dirs[v["key"]]
        tilt.append(float(np.degrees(np.arccos(np.clip(abs(d[2]), 0, 1)))))
        keys.append(v["key"])
    return rgb, obj, sil, feats, tilt, keys


def main(objects=30, stride=6, out="data/tilt_pixels.npz"):
    E_rgb, E_obj, E_sil, F, Y, OBJ = [], [], [], [], [], []
    for obj_id in range(1, objects + 1):
        try:
            rgb, part, sil, feats, tilt, keys = rows_for(obj_id, "tless", "train_primesense", stride)
        except Exception as e:
            print("  object %02d skipped: %s" % (obj_id, str(e)[:60]))
            continue
        if not rgb:
            continue
        def run(images):
            return np.concatenate([embed.embed_images(images[i:i + 16]) for i in range(0, len(images), 16)])
        E_rgb.append(run(rgb))
        E_obj.append(run(part))
        E_sil.append(run(sil))
        F.append(np.array(feats, float))
        Y.append(np.array(tilt, float))
        OBJ.append(np.full(len(tilt), obj_id))
        print("  object %02d  %3d views  tilt %.0f-%.0f deg" % (obj_id, len(tilt), min(tilt), max(tilt)))
    np.savez(os.path.join(ROOT, out), rgb=np.concatenate(E_rgb), obj=np.concatenate(E_obj),
             sil=np.concatenate(E_sil),
             feats=np.concatenate(F), tilt=np.concatenate(Y), part=np.concatenate(OBJ))
    print("\n  wrote %s: %d rows over %d objects" % (out, len(np.concatenate(Y)), len(Y)))


if __name__ == "__main__":
    main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]])
