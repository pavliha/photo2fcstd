import numpy as np
import pytest

from photo2fcstd.trace import elements, fit_circle, fit_ellipse, merge_collinear, outline, primitives, snap_rectilinear, symmetrize


def disc(n=200, r=60, cx=100, cy=100):
    y, x = np.ogrid[:2 * cx, :2 * cy]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2


def rect(w=120, h=60, pad=20):
    m = np.zeros((h + 2 * pad, w + 2 * pad), bool)
    m[pad:pad + h, pad:pad + w] = True
    return m


def test_circle_fit_recovers_radius():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts = np.c_[50 + 30 * np.cos(t), -20 + 30 * np.sin(t)]
    cx, cy, r, rel, aspect = fit_circle(pts)
    assert abs(r - 30) < 0.5 and abs(cx - 50) < 0.5 and abs(cy + 20) < 0.5 and rel < 0.01


def test_ellipse_fit_reports_aspect():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    f = fit_ellipse(np.c_[100 + 60 * np.cos(t), 100 + 30 * np.sin(t)])
    assert abs(f["aspect"] - 0.5) < 0.05 and f["rms"] < 1.0


def test_outline_finds_hole_and_rectangularity():
    m = rect()
    m[35:45, 60:70] = False
    poly, shape = outline(m)
    assert shape["rectangularity"] > 0.9
    assert len(shape["holes"]) == 1
    assert 0.0 < shape["hole_frac"] < 0.1


def test_primitives_turn_a_disc_into_one_circle():
    poly, shape = outline(disc())
    loops = primitives([shape["raw"]], 200)
    assert len(loops) == 1 and loops[0]["type"] == "circle"
    assert abs(loops[0]["r"] - 60) < 2


def test_primitives_turn_a_rectangle_into_four_snapped_lines():
    poly, shape = outline(rect())
    loops = primitives([shape["raw"]], 160)
    loop = loops[0]
    assert loop["type"] == "loop"
    assert len(loop["elements"]) == 4
    assert sorted(loop["kinds"]) == ["H", "H", "V", "V"]


def test_symmetrize_mirrors_a_nearly_symmetric_mask():
    m = rect()
    m[20:26, 20:24] = False
    out, axes = symmetrize(m)
    assert axes
    for axis, name in ((1, "x"), (0, "y")):
        if name in axes:
            assert (out == np.flip(out, axis=axis)).all()


def test_merge_collinear_drops_redundant_points():
    pts = np.array([[0, 0], [50, 0], [100, 0], [100, 50], [0, 50]], float)
    assert len(merge_collinear(pts, 5.0)) == 4


def test_a_rectangle_is_not_a_circle():
    poly, shape = outline(rect())
    assert len(shape["raw"]) >= 12
    loops = primitives([shape["raw"]], 160)
    assert loops[0]["type"] == "loop"


def test_a_disc_is_still_a_circle():
    poly, shape = outline(disc())
    assert primitives([shape["raw"]], 200)[0]["type"] == "circle"


def test_collinear_lines_merge_even_when_the_loop_has_an_arc():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [10, 0]},
           {"type": "line", "p0": [10, 0], "p1": [20, 0]},
           {"type": "line", "p0": [20, 0], "p1": [30, 0]},
           {"type": "arc", "p0": [30, 0], "p1": [30, 10], "cx": 25, "cy": 5, "r": 7.07, "ccw": True},
           {"type": "line", "p0": [30, 10], "p1": [0, 10]},
           {"type": "line", "p0": [0, 10], "p1": [0, 0]}]
    merged = trace.merge_line_elements(els, min_len=1.0)
    assert sum(1 for e in merged if e["type"] == "line") == 3
    assert sum(1 for e in merged if e["type"] == "arc") == 1
    assert merged[0]["p0"] == [0, 0] and merged[0]["p1"] == [30, 0]


def test_merging_never_swallows_a_real_corner():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [30, 0]},
           {"type": "line", "p0": [30, 0], "p1": [30, 20]},
           {"type": "line", "p0": [30, 20], "p1": [0, 20]},
           {"type": "line", "p0": [0, 20], "p1": [0, 0]}]
    assert len(trace.merge_line_elements(els, min_len=1.0)) == 4


def test_a_tiny_segment_is_absorbed():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [30, 0]},
           {"type": "line", "p0": [30, 0], "p1": [30.4, 3]},
           {"type": "line", "p0": [30.4, 3], "p1": [30, 20]},
           {"type": "line", "p0": [30, 20], "p1": [0, 20]},
           {"type": "line", "p0": [0, 20], "p1": [0, 0]}]
    assert len(trace.merge_line_elements(els, min_len=5.0)) == 4


def line_loop(points):
    return [{"type": "line", "p0": list(points[i]), "p1": list(points[(i + 1) % len(points)])}
            for i in range(len(points))]


def test_a_nearly_rectangular_slot_becomes_an_exact_rectangle():
    from photo2fcstd import trace
    els = line_loop([(0, 0), (40, 1), (41, 20), (39, 19.4), (0.6, 20)])
    out = trace.rectangularise(els)
    assert len(out) == 4
    xs = sorted({round(e["p0"][0], 3) for e in out})
    ys = sorted({round(e["p0"][1], 3) for e in out})
    assert len(xs) == 2 and len(ys) == 2


def test_an_l_shape_is_left_alone():
    from photo2fcstd import trace
    els = line_loop([(0, 0), (40, 0), (40, 10), (15, 10), (15, 30), (0, 30)])
    assert len(trace.rectangularise(els)) == len(els)


def test_a_loop_with_an_arc_is_left_alone():
    from photo2fcstd import trace
    els = [{"type": "line", "p0": [0, 0], "p1": [10, 0]},
           {"type": "arc", "p0": [10, 0], "p1": [10, 5], "cx": 10, "cy": 2.5, "r": 2.5, "ccw": True},
           {"type": "line", "p0": [10, 5], "p1": [0, 5]},
           {"type": "line", "p0": [0, 5], "p1": [0, 0]}]
    assert trace.rectangularise(els) is els


def test_rectangularise_keeps_the_winding_direction():
    from photo2fcstd import trace
    ccw = line_loop([(0, 0), (40, 0.5), (40, 20), (0, 20)])
    cw = line_loop([(0, 0), (0, 20), (40, 20), (40, 0.5)])
    a, b = trace.rectangularise(ccw), trace.rectangularise(cw)
    assert trace._signed_area(np.array([e["p0"] for e in a], float)) > 0
    assert trace._signed_area(np.array([e["p0"] for e in b], float)) < 0


def test_symmetrize_mirrors_the_holes_not_just_the_outline():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 9:14] = False
    out, axes = symmetrize(mask)
    assert "x" in axes
    assert not out[12:18, 9:14].any()
    assert not out[12:18, 26:31].any()


def test_symmetrize_leaves_an_asymmetric_part_alone():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:20] = True
    mask[8:12, 20:34] = True
    before = mask.sum()
    out, axes = symmetrize(mask)
    assert axes == [] and out.sum() == before


def test_symmetrize_keeps_a_hole_that_is_already_mirrored():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 9:14] = False
    mask[12:18, 26:31] = False
    out, _ = symmetrize(mask)
    assert not out[12:18, 9:14].any() and not out[12:18, 26:31].any()
    assert out[20:30, 15:25].all()


def test_symmetry_reflects_the_half_that_shows_more_of_the_holes():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[12:18, 8:14] = False
    mask[14:16, 26:30] = False
    out, axes = symmetrize(mask)
    assert "x" in axes
    holes_left = int((~out[12:18, 8:14]).sum())
    holes_right = int((~out[12:18, 26:32]).sum())
    assert holes_left == holes_right and holes_left > 4


def test_reflecting_is_exact_for_odd_and_even_widths():
    from photo2fcstd.trace import symmetrize
    for width in (40, 41):
        mask = np.zeros((40, width), bool)
        mask[5:35, 5:width - 5] = True
        mask[12:18, 9:14] = False
        mask[12:18, width - 14:width - 9] = False
        out, axes = symmetrize(mask)
        assert "x" in axes
        assert (out == np.flip(out, axis=1)).all()


def test_a_near_45_degree_edge_snaps_to_exactly_45():
    from photo2fcstd.trace import snap_angles
    pts = np.array([[0.0, 0.0], [100.0, 96.0], [0.0, 120.0]])
    out = snap_angles(pts)
    d = out[1] - out[0]
    assert abs(np.degrees(np.arctan2(d[1], d[0])) - 45.0) < 0.05


def test_a_near_30_degree_edge_snaps_to_exactly_30():
    from photo2fcstd.trace import snap_angles
    pts = np.array([[0.0, 0.0], [100.0, 60.0], [0.0, 90.0]])
    out = snap_angles(pts)
    d = out[1] - out[0]
    assert abs(np.degrees(np.arctan2(d[1], d[0])) - 30.0) < 0.05


def test_a_deliberate_odd_angle_is_left_alone():
    from photo2fcstd.trace import snap_angles
    angle = lambda p, q: np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
    pts = np.array([[0.0, 0.0], [100.0, 40.4], [0.0, 80.0]])
    before = angle(pts[0], pts[1])
    out = snap_angles(pts)
    assert abs(angle(out[0], out[1]) - before) < 0.01


def test_dominant_frame_finds_the_part_rotation():
    from photo2fcstd.trace import dominant_frame
    box = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 40.0], [0.0, 40.0]])
    t = np.radians(20.0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert abs(dominant_frame(box @ rot.T) - 20.0) < 0.5


def test_snapping_in_a_rotated_frame_squares_the_corners():
    from photo2fcstd.trace import dominant_frame, snap_angles
    t = np.radians(20.0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    box = np.array([[0.0, 0.0], [100.0, 2.5], [97.0, 40.0], [-2.0, 38.0]]) @ rot.T
    frame = dominant_frame(box)
    out = snap_angles(box, tol_deg=7.0, frame=frame)
    angles = []
    for i in range(len(out)):
        d = out[(i + 1) % len(out)] - out[i]
        angles.append(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    turns = [abs(abs(angles[i] - angles[(i + 1) % len(angles)]) - 90.0) for i in range(len(angles))]
    assert sum(1 for x in turns if x < 3.0) >= 2


def test_fit_directions_puts_every_edge_on_a_clean_angle():
    from photo2fcstd.trace import fit_directions
    pts = np.array([[0.0, 0.0], [100.0, 2.5], [97.0, 40.0], [-2.0, 38.0]])
    out = fit_directions(pts)
    n = len(out)
    for i in range(n):
        d = out[(i + 1) % n] - out[i]
        a = np.degrees(np.arctan2(d[1], d[0])) % 15.0
        assert min(a, 15.0 - a) < 0.01


def test_fit_directions_closes_the_loop():
    from photo2fcstd.trace import fit_directions
    pts = np.array([[0.0, 0.0], [100.0, 2.5], [97.0, 40.0], [-2.0, 38.0]])
    out = fit_directions(pts)
    n = len(out)
    total = sum(out[(i + 1) % n] - out[i] for i in range(n))
    assert np.allclose(total, 0.0, atol=1e-6)


def test_fit_directions_refuses_a_fit_that_moves_the_shape_too_far():
    from photo2fcstd.trace import fit_directions
    pts = np.array([[0.0, 0.0], [100.0, 2.5], [97.0, 40.0], [-2.0, 38.0]])
    assert np.allclose(fit_directions(pts, max_shift=0.0), pts)
    assert not np.allclose(fit_directions(pts, max_shift=0.5), pts)


def test_every_loop_shares_the_part_frame():
    from photo2fcstd.trace import dominant_frame, primitives
    t = np.radians(20.0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    box = lambda w, h, dx, dy: np.array([[dx, dy], [dx + w, dy], [dx + w, dy + h], [dx, dy + h]])
    dense = lambda pts: np.array([pts[i] + (pts[(i + 1) % len(pts)] - pts[i]) * f
                                  for i in range(len(pts)) for f in np.linspace(0, 1, 40, endpoint=False)])
    outer = dense(box(200.0, 120.0, 0.0, 0.0)) @ rot.T
    hole = dense(box(40.0, 20.0, 60.0, 40.0)) @ rot.T
    loops = primitives([outer.tolist(), hole.tolist()], 200.0)
    angles = []
    for loop in loops:
        if loop["type"] != "loop":
            continue
        for e in loop["elements"]:
            if e["type"] == "line":
                d = np.subtract(e["p1"], e["p0"])
                angles.append(np.degrees(np.arctan2(d[1], d[0])) % 90.0)
    assert len(angles) >= 6
    spread = [min(abs(a - 20.0), abs(a - 20.0 - 90.0), abs(a - 20.0 + 90.0)) for a in angles]
    assert max(spread) < 3.0, sorted(spread)


def rotate_mask(mask, deg):
    from scipy import ndimage
    return ndimage.rotate(mask.astype(np.uint8), deg, reshape=True, order=0) > 0


def test_a_square_is_not_uprighted_onto_its_diagonal():
    from photo2fcstd.trace import upright_mask
    square = np.zeros((160, 160), bool)
    square[30:130, 30:130] = True
    out, angle = upright_mask(rotate_mask(square, 3.0))
    ys, xs = np.nonzero(out)
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    assert abs(w - h) / max(w, h) < 0.12
    assert min(abs(angle), abs(abs(angle) - 90.0)) < 8.0


def test_an_elongated_part_still_uses_its_long_axis():
    from photo2fcstd.trace import upright_mask
    bar = np.zeros((200, 200), bool)
    bar[60:140, 90:110] = True
    out, _ = upright_mask(rotate_mask(bar, 30.0))
    ys, xs = np.nonzero(out)
    assert (ys.max() - ys.min()) > 2.5 * (xs.max() - xs.min())


def test_a_trapezoid_from_perspective_is_squared_up():
    from photo2fcstd.trace import square_quadrilateral
    els = line_loop([(0.0, 0.0), (100.0, 10.0), (100.0, 70.0), (0.0, 90.0)])
    out = square_quadrilateral(els)
    assert len(out) == 4
    sides = [np.hypot(*np.subtract(e["p1"], e["p0"])) for e in out]
    assert abs(sides[0] - sides[2]) < 1e-6 and abs(sides[1] - sides[3]) < 1e-6
    angles = [np.degrees(np.arctan2(*np.subtract(e["p1"], e["p0"])[::-1])) % 90.0 for e in out]
    assert max(min(a, 90.0 - a) for a in angles) < 1e-6


def test_a_strongly_tapered_wedge_is_left_alone():
    from photo2fcstd.trace import square_quadrilateral
    els = line_loop([(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 20.0)])
    assert square_quadrilateral(els) is els


def test_squaring_only_touches_four_sided_loops():
    from photo2fcstd.trace import square_quadrilateral
    els = line_loop([(0.0, 0.0), (50.0, 0.0), (100.0, 20.0), (100.0, 70.0), (0.0, 90.0)])
    assert square_quadrilateral(els) is els


def test_a_hole_that_shows_the_background_is_carved_back_out():
    from photo2fcstd.trace import recover_holes
    image = np.zeros((200, 200, 3), float)
    image[:, :] = (0.95, 0.95, 0.95)
    image[40:160, 40:160] = (0.15, 0.45, 0.25)
    image[80:120, 80:120] = (0.95, 0.95, 0.95)
    filled = np.zeros((200, 200), bool)
    filled[40:160, 40:160] = True
    out = recover_holes(image, filled)
    assert not out[85:115, 85:115].any()
    assert out[45:75, 45:75].all()


def test_a_part_the_colour_of_the_background_is_not_hollowed_out():
    from photo2fcstd.trace import recover_holes
    image = np.zeros((200, 200, 3), float)
    image[:, :] = (0.95, 0.95, 0.95)
    image[40:160, 40:160] = (0.93, 0.94, 0.95)
    filled = np.zeros((200, 200), bool)
    filled[40:160, 40:160] = True
    out = recover_holes(image, filled, tol=0.02)
    assert out.sum() == filled.sum()


def test_a_dark_pocket_is_left_alone():
    from photo2fcstd.trace import recover_holes
    image = np.zeros((200, 200, 3), float)
    image[:, :] = (0.95, 0.95, 0.95)
    image[40:160, 40:160] = (0.15, 0.45, 0.25)
    image[80:120, 80:120] = (0.02, 0.02, 0.02)
    filled = np.zeros((200, 200), bool)
    filled[40:160, 40:160] = True
    assert recover_holes(image, filled).sum() == filled.sum()


def test_a_disc_keeps_its_own_orientation():
    from photo2fcstd.trace import box_angle
    yy, xx = np.mgrid[0:200, 0:200]
    disc = ((yy - 100) ** 2 + (xx - 100) ** 2) < 80 ** 2
    assert box_angle(disc) is None


def test_a_square_gets_a_box_angle():
    from photo2fcstd.trace import box_angle
    square = np.zeros((200, 200), bool)
    square[50:150, 50:150] = True
    assert box_angle(square) is not None


def test_reflection_slivers_are_not_kept_as_holes():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((200, 200), bool)
    mask[40:160, 40:160] = True
    mask[100:102, 60:63] = False
    out, axes = symmetrize(mask)
    assert axes
    assert out[95:107, 55:70].all()


def test_a_real_hole_survives_the_speck_filter():
    from photo2fcstd.trace import symmetrize
    mask = np.zeros((200, 200), bool)
    mask[40:160, 40:160] = True
    mask[80:120, 60:90] = False
    out, _ = symmetrize(mask)
    assert not out[85:115, 65:85].any()


def toothed(teeth=12, radius=100.0, depth=0.12, n=720):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius * (1.0 + depth * np.sign(np.cos(teeth * a)))
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def test_a_toothed_contour_is_recognised_as_repeated():
    from photo2fcstd.trace import boundary_period
    assert boundary_period(toothed(12)) is not None
    assert boundary_period(toothed(30)) is not None


def test_a_plain_disc_is_not_repeated():
    from photo2fcstd.trace import boundary_period
    a = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    assert boundary_period(np.column_stack([100 * np.cos(a), 100 * np.sin(a)])) is None


def test_a_rectangle_is_not_repeated():
    from photo2fcstd.trace import boundary_period
    pts = []
    for f in np.linspace(0, 1, 200, endpoint=False):
        pts += [(f * 200, 0.0), (200.0, f * 100), (200 - f * 200, 100.0), (0.0, 100 - f * 100)]
    assert boundary_period(np.array(pts)) is None


def test_teeth_survive_into_the_elements():
    from photo2fcstd.trace import elements
    els = elements(toothed(12, n=1440), 200.0)
    assert len(els) >= 20, len(els)


def test_a_scalloped_disc_is_not_called_a_circle():
    from photo2fcstd.trace import primitives
    loops = primitives([toothed(16, radius=120.0, depth=0.12, n=1440).tolist()], 240.0)
    assert loops[0]["type"] == "loop"
    assert len(loops[0]["elements"]) >= 16


def test_a_plain_disc_is_still_a_circle():
    from photo2fcstd.trace import primitives
    a = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    disc = np.column_stack([120 * np.cos(a), 120 * np.sin(a)])
    assert primitives([disc.tolist()], 240.0)[0]["type"] == "circle"


def test_shallow_flutes_still_read_as_a_circle():
    from photo2fcstd.trace import primitives
    a = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    r = 120 * (1 + 0.02 * np.sign(np.cos(24 * a)))
    fluted = np.column_stack([r * np.cos(a), r * np.sin(a)])
    assert primitives([fluted.tolist()], 240.0)[0]["type"] == "circle"


def test_deep_scallops_are_drawn_not_rounded_off():
    from photo2fcstd.trace import primitives
    a = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    r = 120 * (1 + 0.12 * np.sign(np.cos(16 * a)))
    scalloped = np.column_stack([r * np.cos(a), r * np.sin(a)])
    assert primitives([scalloped.tolist()], 240.0)[0]["type"] == "loop"


def test_patch_pooling_halves_the_grid():
    from photo2fcstd.patches import pooled
    grid = np.arange(16 * 16 * 4, dtype=float).reshape(16, 16, 4)
    out = pooled(grid, 2)
    assert out.shape == (8, 8, 4)
    assert out[0, 0, 0] == pytest.approx(grid[0:2, 0:2, 0].mean())


def test_patch_pooling_is_a_noop_at_factor_one():
    from photo2fcstd.patches import pooled
    grid = np.zeros((7, 7, 3))
    assert pooled(grid, 1).shape == (7, 7, 3)


def test_tokens_are_square():
    from photo2fcstd.patches import grid_of
    tokens = np.zeros((197, 8))
    assert grid_of(tokens).shape == (14, 14, 8)


def test_a_wire_tail_is_trimmed_off_the_part():
    from photo2fcstd.trace import trim_appendages
    mask = np.zeros((200, 200), bool)
    mask[60:140, 40:140] = True
    mask[95:101, 140:195] = True
    out = trim_appendages(mask, width_frac=0.09)
    assert out[70:130, 50:130].all()
    assert not out[95:101, 160:195].any()


def test_a_part_that_is_thin_all_over_is_left_alone():
    from photo2fcstd.trace import trim_appendages
    mask = np.zeros((200, 200), bool)
    mask[98:104, 20:180] = True
    assert trim_appendages(mask, width_frac=0.09).sum() == mask.sum()


def test_trimming_keeps_the_outline_of_what_survives():
    from photo2fcstd.trace import trim_appendages
    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True
    mask[95:100, 150:190] = True
    out = trim_appendages(mask, width_frac=0.09)
    assert out[50:150, 50:150].all()


def test_appendage_trimming_is_off_by_default():
    from photo2fcstd.trace import TRIM_APPENDAGE, trim_appendages
    assert TRIM_APPENDAGE == 0.0
    mask = np.zeros((200, 200), bool)
    mask[60:140, 40:140] = True
    mask[95:101, 140:195] = True
    assert trim_appendages(mask).sum() == mask.sum()


def test_a_split_arc_is_refit_as_one_arc_and_corners_are_not():
    from photo2fcstd.trace import chain_arcs
    line_el = lambda run: {"type": "line", "p0": run[0].tolist(), "p1": run[-1].tolist(), "_run": run}
    seg = lambda p, q, k=30: np.linspace(p, q, k)
    edges = np.linspace(0, np.radians(60), 4)
    pieces = [np.c_[120 * np.cos(t), 120 * np.sin(t)]
              for t in (np.linspace(a, b, 40) for a, b in zip(edges, edges[1:]))]
    p_end, p_start = pieces[-1][-1], pieces[0][0]
    box = [seg(p_end, p_end + [-260, 0]), seg(p_end + [-260, 0], [p_start[0] - 260, p_start[1]]),
           seg([p_start[0] - 260, p_start[1]], p_start)]
    out = chain_arcs([line_el(r) for r in pieces + box], 300.0)
    kinds = [e["type"] for e in out]
    assert kinds.count("arc") == 1 and kinds.count("line") == 3, kinds
    arc = next(e for e in out if e["type"] == "arc")
    assert abs(arc["r"] - 120.0) < 2.0

    square = [seg([0, 0], [100, 0]), seg([100, 0], [100, 100]), seg([100, 100], [0, 100]), seg([0, 100], [0, 0])]
    assert all(e["type"] == "line" for e in chain_arcs([line_el(r) for r in square], 100.0))

    shallow = [seg([0, 0], [100, 12]), seg([100, 12], [200, 0]), seg([200, 0], [200, -80]),
               seg([200, -80], [0, -80]), seg([0, -80], [0, 0])]
    assert all(e["type"] == "line" for e in chain_arcs([line_el(r) for r in shallow], 200.0))


def test_a_freeform_boundary_merges_to_one_spline_and_fillets_do_not(monkeypatch):
    import photo2fcstd.trace as trace_mod
    monkeypatch.setattr(trace_mod, "TANGENT_MERGE", True)
    from photo2fcstd.trace import elements
    def blob(n=1000):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 260 + 50 * np.sin(2 * t) + 35 * np.sin(3 * t + 1.1)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    els = elements(blob(), 500.0)
    kinds = [e["type"] for e in els]
    assert "bsplinecurve" in kinds or kinds == ["bsplinecurve"], kinds

    def rounded_rect(w=300.0, h=180.0, r=30.0, k=28):
        import numpy as np
        segs = []
        cs = [(w - r, h - r, 0), (r, h - r, 90), (r, r, 180), (w - r, r, 270)]
        for i, (cx, cy, a0) in enumerate(cs):
            a = np.radians(np.linspace(a0, a0 + 90, k))
            segs.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
        out = []
        for i in range(4):
            out.append(segs[i])
            p, q = segs[i][-1], segs[(i + 1) % 4][0]
            out.append(np.linspace(p, q, 40)[1:-1])
        return np.vstack(out)
    els = elements(rounded_rect(), 300.0)
    kinds = [e["type"] for e in els]
    assert "bsplinecurve" not in kinds, kinds
