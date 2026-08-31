import numpy as np

from photo2fcstd import axis_model, carve


def plate(thin_axis):
    n = [40, 40, 40]
    n[thin_axis] = 4
    g = np.mgrid[0:n[0], 0:n[1], 0:n[2]]
    return {"points_mm": np.column_stack([a.ravel() for a in g]).astype(float), "voxel_mm": 1.0}


def test_features_are_finite_and_per_axis():
    f = axis_model.features(plate(1))
    assert f.shape == (3, 12)
    assert np.isfinite(f).all()
    assert not np.allclose(f[0], f[1])


def test_plate_projects_along_its_thin_axis():
    for a in (0, 1, 2):
        assert carve.base_axis(plate(a)) == a


def test_falls_back_without_a_model(monkeypatch):
    monkeypatch.setattr(axis_model, "_CACHE", {"m": None})
    assert carve.base_axis(plate(2)) == 2


def test_a_cube_scores_all_three_axes_close():
    import joblib
    n = 30
    g = np.mgrid[0:n, 0:n, 0:n]
    cube = {"points_mm": np.column_stack([a.ravel() for a in g]).astype(float), "voxel_mm": 1.0}
    p = joblib.load("data/axis_model.joblib").predict_proba(axis_model.features(cube))[:, 1]
    assert p.max() - p.min() < 0.15
