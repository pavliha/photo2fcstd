import numpy as np
import trimesh

from photo2fcstd.overlay import compare, grid
from photo2fcstd.score import PITCH, best_alignment, best_iou, signed_perms


def test_signed_perms_are_the_24_rotations():
    perms = signed_perms()
    assert len(perms) == 24
    assert len({(p, s) for p, s in perms}) == 24


def test_rotated_box_scores_one():
    box = trimesh.creation.box((2, 1, 0.5))
    value, raw = best_iou(box, trimesh.creation.box((1, 2, 0.5)))
    assert value > 0.97 and raw <= value


def test_cylinder_is_not_a_box():
    value, _ = best_iou(trimesh.creation.box((2, 1, 0.5)), trimesh.creation.cylinder(0.5, 2.0))
    assert value < 0.9


def test_grid_keeps_every_voxel(tmp_path):
    box = trimesh.creation.box((2, 1, 0.5))
    _, _, _, ref, _ = best_alignment(box, trimesh.creation.box((2, 1, 0.5)), PITCH)
    assert grid(ref, PITCH).sum() == len(ref)


def test_compare_reports_no_error_for_identical_meshes(tmp_path):
    p = tmp_path / "box.stl"
    trimesh.creation.box((2, 1, 0.5)).export(p)
    c = compare(str(p), str(p), PITCH)
    assert c["iou3d"] > 0.99 and c["missing3d"] < 0.01 and c["extra3d"] < 0.01
    assert c["views"]["face"]["iou"] > 0.99
