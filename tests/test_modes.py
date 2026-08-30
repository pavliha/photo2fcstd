import pytest

from photo2fcstd import analysis, modes

CASES = [("01407", "revolve"), ("00359", "revolve"), ("01289", "plan"), ("00476", "plan"),
         ("00171", "profile"), ("00133", "profile"), ("00523", "stations"), ("01745", "stations")]


@pytest.mark.parametrize("part,expected", CASES)
def test_mode_selection(dataset, photos_of, part, expected):
    views = [analysis.view(p) for p in photos_of(part)[:3]]
    mode, src = modes.select(views)
    assert mode == expected
    assert src in views


def test_forced_mode_overrides(dataset, photos_of):
    views = [analysis.view(p) for p in photos_of("01289")[:3]]
    assert modes.select(views, "stations")[0] == "stations"
    assert modes.select(views, "revolve")[0] == "revolve"


def test_same_face_detects_a_plate(dataset, photos_of):
    assert modes.same_face([analysis.view(p) for p in photos_of("01289")[:3]])
    assert not modes.same_face([analysis.view(p) for p in photos_of("00523")[:3]])
