import numpy as np
from photo2fcstd.sketch_score import chain_edges, polygon_of


def test_edges_out_of_order_still_make_the_right_ring():
    # a thin U, given with its edges shuffled and some reversed
    ring = [[(0, 0), (0, 10)], [(0, 10), (2, 10)], [(2, 10), (2, 2)], [(2, 2), (8, 2)],
            [(8, 2), (8, 10)], [(8, 10), (10, 10)], [(10, 10), (10, 0)], [(10, 0), (0, 0)]]
    shuffled = [ring[3], list(reversed(ring[0])), ring[6], ring[1],
                list(reversed(ring[4])), ring[7], ring[2], list(reversed(ring[5]))]
    good = polygon_of([chain_edges(ring)])
    scrambled_fixed = polygon_of([chain_edges(shuffled)])
    assert abs(good.area - scrambled_fixed.area) < 1e-6, (good.area, scrambled_fixed.area)
    # and it is the thin U, not the filled rectangle
    assert good.area < 0.6 * 100
