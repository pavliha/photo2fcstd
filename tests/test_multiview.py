import math

import numpy as np
import torch

from photo2fcstd import multiview as mv


def _render(P, depth, rvec, t, f, c):
    G = mv._grid()
    soft = mv.silhouette(mv._T([P]), mv._T([depth]), mv._T([rvec]), mv._T([t]), mv._T([f]), mv._T([c]), G)
    return (soft > 0.5).cpu().numpy().reshape(mv.RES, mv.RES)


def _truth():
    return np.array([[-0.5, -0.4], [0.5, -0.4], [0.5, -0.1], [-0.1, -0.1], [-0.1, 0.4], [-0.5, 0.4]])


def test_three_tilted_views_recover_the_face():
    P, depth = _truth(), 0.15
    f, c = 300.0, np.array([mv.RES / 2, mv.RES / 2])
    poses = [(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 2.6])),
             (np.array([0.0, -0.45, 0.0]), np.array([0.0, 0.0, 2.6])),
             (np.array([0.3, 0.3, 0.0]), np.array([0.0, 0.0, 2.6]))]
    masks = [_render(P, depth, r, t, f, c) for r, t in poses]
    f35 = 36.0 * f / mv.RES
    res = mv.fit(masks, [f35] * 3, ref=0, iters=200)
    assert min(res["ious"]) > 0.95, res["ious"]
    truth_tilt = [math.degrees(np.linalg.norm(r)) for r, _ in poses]
    assert all(abs(a - b) < 6 for a, b in zip(res["tilts"], truth_tilt)), (res["tilts"], truth_tilt)
    assert abs(res["depth"] - depth) < 0.06, res["depth"]
    rect, _ = mv.rectified_mask(res, masks[0])
    from photo2fcstd.design import _aligned_iou
    img = np.zeros((mv.RES, mv.RES), np.uint8)
    import cv2
    cv2.fillPoly(img, [((P + 0.5) * (mv.RES - 40) + 20).astype(np.int32)], 1)
    from photo2fcstd.design import _canon
    a, b = _canon(rect), _canon(img > 0)
    best = max(float(np.logical_and(a, bb).sum() / np.logical_or(a, bb).sum())
               for bb in (b, b[::-1], b[:, ::-1], np.rot90(b), np.rot90(b)[::-1], np.rot90(b, 2), np.rot90(b, 3), np.rot90(b, 3)[::-1]))
    assert best > 0.93, best
