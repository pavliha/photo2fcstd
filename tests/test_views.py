from photo2fcstd import views


def view(name, elongation, rect=0.9, sol=0.95, holes=0.0):
    return {"source": "/p/%s.jpg" % name, "elongation": elongation,
            "select": {"elongation": elongation, "rectangularity": rect,
                       "solidity": sol, "hole_frac": holes}}


def test_fewer_than_three_are_all_kept():
    offered = [view("a", 1.0), view("b", 2.0)]
    assert views.spread(offered) == offered


def test_the_flattest_view_is_always_kept():
    offered = [view("tall", 9.0), view("flat", 1.1), view("mid", 4.0), view("other", 3.0)]
    assert any(v["source"].endswith("flat.jpg") for v in views.spread(offered))


def test_it_picks_views_that_differ_rather_than_the_first_three():
    offered = [view("a", 1.10), view("b", 1.11), view("c", 1.12), view("d", 6.0, rect=0.4)]
    chosen = [v["source"] for v in views.spread(offered)]
    assert "/p/d.jpg" in chosen


def test_exactly_three_are_returned_from_many():
    offered = [view(str(i), 1.0 + i * 0.7) for i in range(9)]
    assert len(views.spread(offered)) == 3


def test_the_note_names_what_was_used_and_stays_quiet_otherwise():
    offered = [view("a", 1.0), view("b", 2.0), view("c", 3.0), view("d", 4.0)]
    chosen = views.spread(offered)
    assert "4 photographs" in views.describe(chosen, offered)
    assert views.describe(offered[:2], offered[:2]) == ""


def test_a_view_without_selection_statistics_still_sorts():
    plain = {"source": "/p/x.jpg", "elongation": 2.0,
             "shape": {"rectangularity": 0.8, "solidity": 0.9, "hole_frac": 0.0}}
    offered = [plain, view("b", 5.0), view("c", 1.0), view("d", 3.0)]
    assert len(views.spread(offered)) == 3
