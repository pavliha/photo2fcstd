import numpy as np
import pytest

from photo2fcstd import cornernet, synth, trace


def square(n=400, side=100.0):
    t = np.linspace(0, 4, n, endpoint=False)
    e = np.floor(t).astype(int)
    f = (t - e)[:, None]
    c = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], float)
    return c[e] + f * (c[e + 1] - c[e])


def test_peaks_decode_between_points():
    pts = square()
    h = np.zeros(len(pts))
    for i in (0, 100, 200, 300):
        for d in range(-4, 5):
            h[(i + d) % len(pts)] = max(h[(i + d) % len(pts)], np.exp(-0.5 * (d / 2.0) ** 2))
    found = cornernet.peaks(h, pts)
    assert len(found) == 4
    truth = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert np.linalg.norm(found[:, None] - truth[None], axis=2).min(axis=1).max() < 2.0


def test_a_circle_has_no_corners():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    ring = np.column_stack([np.cos(t), np.sin(t)]) * 100 + 150
    assert cornernet.peaks(np.zeros(len(ring)), ring).shape == (0, 2)


def test_the_heatmap_target_peaks_at_a_real_corner():
    loops = [[("line", np.array([[0.0, 0.0], [100.0, 0.0]])),
              ("line", np.array([[100.0, 0.0], [100.0, 100.0]]))]]
    heat, corners = synth.corner_heatmap(square(), loops)
    assert len(corners) == 2
    assert heat.max() > 0.9


def test_learned_corner_paths_are_off_by_default():
    assert not trace.LEARNED_CORNERS
    assert not trace.FILTERED_CORNERS
