"""CPU tests of the v15 E5gen pieces: the held-out grids and the paired verdict."""

import math

from Spin_Colour_Kernel import width_law_b as W


def test_grids_gen_are_held_out_and_cover_both_sweeps():
    g = W.grids_gen(threshold=221184)
    L1s = {x[4][0] for x in g if x[0] in ("layer", "layer-heis")}
    assert L1s == set(W.GEN_L1)
    assert not L1s & set(W.CONF_L1)
    assert {x[0] for x in g} == {"below", "above", "layer", "layer-heis"}


def _row(factor, L1, xla, gen):
    return {"factor": factor, "B": 21, "n": 3, "b": 1, "box": [L1, 31, 31],
            "xla_ns_per_update": xla, "generated_ns_per_update": gen, "generated_periodic": L1 % 2 == 0}


def test_gen_verdict_speedup_and_spread():
    rows = [_row("layer-heis", 100, [4.0, 4.1, 3.9, 4.0], [0.5, 0.5, 0.52, 0.49]),
            _row("layer-heis", 101, [6.0, 6.2, 5.9, 6.1], [0.6, 0.61, 0.59, 0.6])]
    v = W.gen_verdict(rows, boots=200)
    assert v["all_faster"]
    assert math.isclose(v["grids"][0]["log_speedup"], sum(math.log(x / g) for x, g in
                        zip(rows[0]["xla_ns_per_update"], rows[0]["generated_ns_per_update"])) / 4)
    s = v["spread"]["layer-heis"]
    assert math.isclose(s["xla_max_over_min"], 6.05 / 4.0)
    assert s["generated_max_over_min"] < s["xla_max_over_min"]


def test_gen_verdict_flags_a_slower_grid():
    rows = [_row("layer", 100, [1.0, 1.0, 1.0], [2.0, 2.1, 1.9])]
    v = W.gen_verdict(rows, boots=200)
    assert not v["all_faster"] and not v["grids"][0]["generated_faster"]


def test_grids_genb_same_geometry_varied_replicas():
    g = W.grids_genb()
    assert {x[4] for x in g} == {W.GENB_BOX}
    assert {x[1] for x in g} == set(W.GENB_B)
    assert {x[2] for x in g} == {1, 3}


def test_genb_verdict_counts_upward_jumps():
    def r(B, x, g):
        return {"factor": "B-heis", "B": B, "n": 3, "b": 1, "box": [27, 27, 27],
                "xla_ns_per_update": [x] * 3, "generated_ns_per_update": [g] * 3}
    rows = [r(1, 10.0, 1.0), r(2, 6.0, 0.6), r(3, 7.0, 0.5), r(4, 5.0, 0.45)]
    v = W.genb_verdict(rows, boots=100)
    m = v["per_model"]["B-heis"]
    assert m["xla_upward_jumps"] == 1 and m["generated_upward_jumps"] == 0
    assert v["generated_smooth"] and v["generated_faster_everywhere"]
