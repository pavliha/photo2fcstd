import pytest

from photo2fcstd import measured


def test_a_plain_measurement_parses():
    got = measured.parse("body_width=12.53")
    assert got == {"name": "body_width", "mm": 12.53, "source": None}


def test_units_and_a_source_photo_are_allowed():
    got = measured.parse("lens_diameter = 16.15 mm @ /tmp/IMG_3151.jpg")
    assert got["name"] == "lens_diameter" and got["mm"] == 16.15
    assert got["source"].endswith("IMG_3151.jpg")


def test_nonsense_is_refused_rather_than_guessed():
    assert measured.parse("12.53") is None
    assert measured.parse("width=") is None
    with pytest.raises(ValueError) as err:
        measured.parse_all(["width=12.5", "nonsense"])
    assert "nonsense" in str(err.value)


def test_measuring_the_same_name_twice_is_refused():
    with pytest.raises(ValueError) as err:
        measured.parse_all(["w=1.0", "w=2.0"])
    assert "measured twice" in str(err.value)


def test_rows_carry_the_label_and_the_photo():
    rows = measured.rows(measured.parse_all(["overall_length=21.72@IMG_3153.jpg", "gap=2.51"]))
    assert rows[0] == ("overall_length", 21.72, "measured, IMG_3153.jpg")
    assert rows[1] == ("gap", 2.51, "measured, caliper")


def test_a_named_measurement_sets_the_scale():
    m = measured.parse_all(["overall_length=21.72"])
    assert measured.scale_from(m, "overall_length", 1086.0) == pytest.approx(0.02)
    assert measured.scale_from(m, "missing", 1086.0) is None


def test_the_document_carries_the_readings_through_to_the_sheet():
    import sys
    from photo2fcstd import measured
    spec = {"measured": [{"name": "overall_length", "mm": 21.72, "source": "IMG_3153.jpg"}]}
    rows = measured.rows(spec["measured"])
    assert rows == [("overall_length", 21.72, "measured, IMG_3153.jpg")]


def test_no_readings_means_no_rows():
    from photo2fcstd import measured
    assert measured.rows([]) == []
