import cv2
import numpy as np
import pytest
import trimesh

from photo2fcstd.carve import carve, mesh_of, project, spec_from_carve


def views_of(target_mm, n=12, radius=160.0):
    K = np.array([[1800.0, 0, 640.0], [0, 1800.0, 480.0], [0, 0, 1.0]])
    out = []
    for i in range(n):
        a = np.radians(i * 360.0 / n)
        t = np.radians(45 + 35 * (i % 3))
        eye = np.array([target_mm[0] + radius * np.cos(a) * np.sin(t),
                        target_mm[1] + radius * np.sin(a) * np.sin(t), radius * np.cos(t)])
        forward = np.array(target_mm) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rvec, _ = cv2.Rodrigues(np.stack([right, down, forward]))
        out.append({"rvec": rvec, "tvec": (-np.stack([right, down, forward]) @ eye).reshape(3, 1),
                    "K": K, "dist": np.zeros(5)})
    return out


def silhouettes(mesh, views, shape=(960, 1280)):
    masks = []
    for v in views:
        uv = project(mesh.vertices, v).astype(np.float32)
        img = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(img, cv2.convexHull(uv.reshape(-1, 1, 2)).astype(np.int32), 1)
        masks.append(img.astype(bool))
    return masks


def carved_box(size=(20.0, 12.0, 6.0), voxel=0.5):
    mesh = trimesh.creation.box(size)
    mesh.apply_translation([40.0, 40.0, size[2] / 2])
    views = views_of([40.0, 40.0, size[2] / 2])
    return carve(views, silhouettes(mesh, views), voxel_mm=voxel, allow_misses=0,
                 bounds=((20, 65), (20, 65), (0, 25))), mesh


def test_carving_recovers_metric_extents():
    carved, mesh = carved_box()
    error = np.sort(carved["extents_mm"]) - np.sort(mesh.extents)
    assert (error >= -0.6).all(), error
    assert (error <= 2.0).all(), error


def test_the_visual_hull_contains_the_part():
    carved, mesh = carved_box()
    assert carved["volume_mm3"] >= mesh.volume * 0.95
    assert carved["volume_mm3"] <= mesh.volume * 1.6


def test_more_views_tighten_the_hull():
    from test_carve import views_of as vo
    import test_carve
    sizes = {}
    for n in (6, 16):
        test_carve.views_of = lambda t, _n=n, radius=160.0: vo(t, _n, radius)
        carved, mesh = carved_box()
        sizes[n] = float(np.sum(np.sort(carved["extents_mm"]) - np.sort(mesh.extents)))
    test_carve.views_of = vo
    assert sizes[16] <= sizes[6]


def test_spec_from_carve_measures_the_height():
    carved, mesh = carved_box(size=(20.0, 12.0, 3.0))
    doc = spec_from_carve(carved, "box")
    assert doc["mm_per_px"] == 1.0
    assert doc["outline"]["depth_px"] == pytest.approx(3.0, abs=1.5)
    assert "not guessed" in doc["outline"]["depth_note"]
    assert doc["outline"]["loops"]


def test_carved_mesh_is_a_solid():
    carved, _ = carved_box(voxel=1.0)
    assert mesh_of(carved).volume > 0


def test_carved_mesh_is_watertight_with_positive_volume():
    carved, truth = carved_box(voxel=1.0)
    mesh = mesh_of(carved)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert mesh.volume == pytest.approx(truth.volume, rel=0.5)


def test_debias_height_trims_what_a_low_camera_cannot_see():
    from photo2fcstd.carve import debias_height
    xs, ys, zs = np.mgrid[0:40, 0:20, 0:30]
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]).astype(float)
    carved = {"points_mm": pts, "voxel_mm": 1.0, "extents_mm": np.ptp(pts, axis=0) + 1.0,
              "min_elevation_deg": 15.0}
    out = debias_height(carved)
    cut = 19.0 / 2 * np.tan(np.radians(15.0))
    assert out["height_debias_mm"] == pytest.approx(cut, abs=0.5)
    assert np.ptp(out["points_mm"][:, 2]) < np.ptp(pts[:, 2])


def test_debias_height_is_a_noop_without_an_elevation():
    from photo2fcstd.carve import debias_height
    pts = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T.astype(float)
    carved = {"points_mm": pts, "voxel_mm": 1.0}
    assert debias_height(carved) is carved
