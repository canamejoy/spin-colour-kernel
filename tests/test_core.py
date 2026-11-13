"""Invariants of the colourings (7)-(8) and of the generated kernel, on the Triton CPU interpreter."""

import numpy as np
import pytest
import torch

from Spin_Colour_Kernel import colouring as co
from Spin_Colour_Kernel import crystals as cr
from Spin_Colour_Kernel import system as sy
from Spin_Colour_Kernel.sampler import Sampler

EXPECTED = {  # crystal, shells -> (linear colours, clique number); E3 / H2
    ("square", (1,)): (2, 2), ("sc", (1,)): (2, 2), ("sc", (1, 2)): (4, 4), ("sc", (1, 2, 3, 4)): (9, 8),
    ("bcc", (1,)): (2, 2), ("fcc", (1,)): (4, 4), ("triangular", (1,)): (3, 3),
    ("honeycomb", (1,)): (2, 2), ("kagome", (1,)): (3, 3),
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_colouring_counts_and_admissibility(key):
    c = cr.make(*key)
    col = co.search(c)
    assert co.admissible(col, c)
    assert (col.n_colours, co.clique_number(c)) == EXPECTED[key]


def test_bond_counts():
    assert [len(cr.make(n).bonds) for n in ("square", "sc", "bcc", "fcc", "triangular", "kagome")] == [4, 6, 16, 48, 6, 12]


def test_seam_condition_rejects_odd_box():
    c = cr.make("triangular")
    col = co.search(c)
    assert not co.seam_ok(col, (4, 4), (True, True))
    with pytest.raises(ValueError):
        sy.System("bad", 1, c, (4, 4), (True, True), {1: 1.0}, colouring=col)


SMALL = [sy.ising("square", 4), sy.xy("square", 4), sy.heisenberg("bcc", 2),
         sy.heisenberg("kagome", 3, J=-1.0), sy.nanodot(Rd=2.0, L=2)]


@pytest.mark.parametrize("s", SMALL, ids=lambda s: s.name)
def test_parallel_equals_sequential_reference(s):
    A, R = Sampler(s, 2, device="cpu", seed=11), Sampler(s, 2, device="cpu", seed=11)
    A.set_temperature(1.5)
    R.set_temperature(1.5)
    S1 = A.init_spins(seed=2)
    S2 = S1.clone()
    for _ in range(2):
        A.sweep(S1)
        R.reference_sweep(S2)
        assert torch.equal(S1, S2)
    assert not torch.equal(S1, A.init_spins(seed=2))


@pytest.mark.parametrize("s", SMALL, ids=lambda s: s.name)
def test_zero_temperature_never_raises_energy(s):
    """At beta -> infinity only dE <= 0 is accepted, so the torch energy cannot rise: checks the local field sign."""
    A = Sampler(s, 3, device="cpu", seed=5)
    A.set_temperature(1e-6)
    S = A.init_spins(seed=1)
    E = A.energy(S)
    for _ in range(3):
        A.sweep(S)
        E2 = A.energy(S)
        assert torch.all(E2 <= E + 1e-4)
        E = E2


def test_masked_sites_stay_zero_and_unit_norm_elsewhere():
    s = sy.nanodot(Rd=2.0, L=2)
    A = Sampler(s, 2, device="cpu", seed=3)
    S = A.init_spins(seed=0)
    A.sweep(S, 2)
    m = torch.as_tensor(s.site_mask())
    norms = S.norm(dim=1)
    assert torch.all(norms[:, ~m] == 0)
    assert torch.allclose(norms[:, m], torch.ones_like(norms[:, m]), atol=1e-5)


def test_barker_acceptance_keeps_exactness():
    s = sy.ising("square", 4)
    s.acceptance = "barker"
    A, R = Sampler(s, 2, device="cpu", seed=4), Sampler(s, 2, device="cpu", seed=4)
    S1 = A.init_spins(seed=1)
    S2 = S1.clone()
    A.sweep(S1, 2)
    R.reference_sweep(S2, 2)
    assert torch.equal(S1, S2)


MANY = [sy.System("Ising sc odd periodic", 1, cr.make("sc"), (5, 5, 5), (True,) * 3, {1: 1.0}),
        sy.System("Heisenberg sc 6x5x5 periodic", 3, cr.make("sc"), (6, 5, 5), (True,) * 3, {1: 1.0}),
        sy.heisenberg("fcc", 2),
        sy.System("XY sc 10x5x5 periodic", 2, cr.make("sc"), (10, 5, 5), (True,) * 3, {1: 1.0})]


@pytest.mark.parametrize("s", MANY, ids=lambda s: s.name)
def test_compacted_sweep_equals_reference_and_full_rows(s):
    """Colour compaction (STRIDE = x-period) changes which programs run, never a spin."""
    A = Sampler(s, 2, device="cpu", seed=3)
    F = Sampler(s, 2, device="cpu", seed=3, compact=False)
    R = Sampler(s, 2, device="cpu", seed=3)
    for X in (A, F, R):
        X.set_temperature(1.2)
    S1 = A.init_spins(seed=4)
    S2, S3 = S1.clone(), S1.clone()
    for _ in range(2):
        A.sweep(S1)
        F.sweep(S2)
        R.reference_sweep(S3)
        assert torch.equal(S1, S2)
        assert torch.equal(S1, S3)
    assert not torch.equal(S1, A.init_spins(seed=4))


def test_x_period_of_odd_periodic_box():
    from Spin_Colour_Kernel.generator import x_period
    s = MANY[0]
    assert s.colouring.n_colours == 5 and x_period(s.colouring) == 5
    assert Sampler(s, 1, device="cpu").stride == 5


def test_compacted_tile_spans_several_rows():
    s = MANY[0]                         # Z5 on 5^3: one site per row and colour, 25 rows per program
    A = Sampler(s, 1, device="cpu")
    assert (A.bx, A.rows_per) == (1, 25)
    T = Sampler(MANY[3], 1, device="cpu")
    assert T.stride == 5 and T.bx == 2 and T.rows_per == 25


ISING = [sy.ising("square", 4), sy.System("Ising sc odd periodic", 1, cr.make("sc"), (5, 5, 5), (True,) * 3, {1: 1.0})]


@pytest.mark.parametrize("s", ISING, ids=lambda s: s.name)
def test_ising_int8_storage_matches_float32_chain(s):
    """int8 storage gives the same chain as float32 storage, site by site."""
    A = Sampler(s, 2, device="cpu", seed=7, packed=False)
    F = Sampler(s, 2, device="cpu", seed=7, packed=False, spin_dtype=torch.float32)
    for X in (A, F):
        X.set_temperature(2.5)
    S8, S32 = A.init_spins(seed=3), F.init_spins(seed=3)
    assert S8.dtype == torch.int8 and S32.dtype == torch.float32
    assert torch.equal(S8.float(), S32)
    A.sweep(S8, 3)
    F.sweep(S32, 3)
    assert torch.equal(S8.float(), S32)
    assert torch.allclose(A.energy(S8), F.energy(S32))


def test_vector_models_keep_float32_storage():
    assert Sampler(sy.xy("square", 4), 1, device="cpu").init_spins().dtype == torch.float32
    assert Sampler(sy.heisenberg("bcc", 2), 1, device="cpu").init_spins().dtype == torch.float32


PACKED = ISING + [sy.System("Ising sc open", 1, cr.make("sc"), (4, 3, 3), (False,) * 3, {1: 1.0}),
                  sy.ising("triangular", 6), sy.ising("square", 80),
                  sy.System("Ising sc long rows", 1, cr.make("sc"), (325, 5, 1), (True, True, False), {1: 1.0})]


@pytest.mark.parametrize("s", PACKED, ids=lambda s: f"{s.name} {s.box}")
def test_bit_packed_ising_matches_int8_chain(s):
    """Ising defaults to 32 replicas per int32 word; every replica follows the int8 chain exactly,
    including a partial last word (B = 40), compacted tiles, open boundaries and rows split into chunks."""
    B = 40
    P = Sampler(s, B, device="cpu", seed=5)
    I8 = Sampler(s, B, device="cpu", seed=5, packed=False)
    assert P.packed and not I8.packed
    T = torch.linspace(1.5, 4.0, B)
    for X in (P, I8):
        X.set_temperature(T)
    Sp, S8 = P.init_spins(seed=9), I8.init_spins(seed=9)
    assert Sp.dtype == torch.int32 and Sp.shape[0] == 2
    assert torch.equal(P.unpack(Sp), S8)
    P.sweep(Sp, 2)
    I8.sweep(S8, 2)
    assert torch.equal(P.unpack(Sp), S8)
    assert torch.allclose(P.energy(Sp), I8.energy(S8))
    assert torch.allclose(P.magnetization(Sp), I8.magnetization(S8))


def test_bit_packed_reference_and_sync_paths():
    s = sy.ising("square", 4)
    A, R = Sampler(s, 33, device="cpu", seed=3), Sampler(s, 33, device="cpu", seed=3)
    for X in (A, R):
        X.set_temperature(2.2)
    S1 = A.init_spins(seed=1)
    S2 = S1.clone()
    A.sweep(S1, 2)
    R.reference_sweep(S2, 2)
    assert torch.equal(S1, S2)
    Y = Sampler(s, 33, device="cpu", seed=3, packed=False)
    Y.set_temperature(2.2)
    Sy = Y.init_spins(seed=1)
    S3 = A.init_spins(seed=1)
    A.set_seed(3)
    A.sync_sweep(S3, 2)
    Y.sync_sweep(Sy, 2)
    assert torch.equal(A.unpack(S3), Sy)


def test_bit_packed_long_compacted_rows_use_chunks():
    s = sy.System("Ising sc long rows", 1, cr.make("sc"), (325, 5, 1), (True, True, False), {1: 1.0})
    P = Sampler(s, 40, device="cpu")
    assert (P.stride, P.bx, P.rows_per, P.pchunks) == (5, 65, 1, 2)


def test_ising_packs_only_when_the_batch_fills_a_word():
    s = sy.ising("square", 4)
    assert not Sampler(s, 16, device="cpu").packed
    assert Sampler(s, 32, device="cpu").packed
    assert Sampler(s, 16, device="cpu").init_spins().dtype == torch.int8


# ------------------------------------------------------------------ E4 analysis: multi-histogram reweighting
def _gamma_samples(betas, k=40.0, n=4000, seed=0):
    """Energies from g(E) ~ E^(k-1): at inverse temperature b, E ~ Gamma(k, 1/b), so <E> = k/b, var = k/b^2."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.gamma(k, 1.0 / b, size=n) for b in betas])     # (temps, samples)


def test_multi_histogram_recovers_exact_moments_between_temperatures():
    from Spin_Colour_Kernel.exp_perf import multi_histogram
    k = 40.0
    betas = np.array([0.9, 1.0, 1.1])
    E = _gamma_samples(betas, k)
    grid_b = np.linspace(0.9, 1.1, 9)
    mean_E, mean_E2 = multi_histogram(E, [E, E * E], 1.0 / betas, 1.0 / grid_b)
    assert np.allclose(mean_E, k / grid_b, rtol=0.01)
    assert np.allclose(mean_E2, k * (k + 1) / grid_b ** 2, rtol=0.02)


def test_multi_histogram_bridges_gaps_that_single_histogram_cannot():
    """Temperatures far apart: the nearest-temperature reweighting has no samples midway; the combined one does."""
    from Spin_Colour_Kernel.exp_perf import multi_histogram
    k = 400.0                                   # narrow histograms: relative width 1/sqrt(k) = 5 %
    betas = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    E = _gamma_samples(betas, k, n=20000, seed=1)
    grid_b = np.linspace(0.9, 1.1, 41)
    (mean_E,) = multi_histogram(E, [E], 1.0 / betas, 1.0 / grid_b)
    assert np.max(np.abs(mean_E * grid_b / k - 1)) < 0.003


def test_multi_histogram_sample_weights_equal_repeated_samples():
    """Bootstrap weights (multiplicities) must give the same answer as physically repeating those samples."""
    from Spin_Colour_Kernel.exp_perf import multi_histogram
    betas = np.array([0.95, 1.05])
    E = _gamma_samples(betas, 60.0, n=500, seed=2)
    w = np.zeros_like(E)
    w[:, :250] = 2.0                            # first half counted twice, second half dropped
    rep = np.concatenate([E[:, :250], E[:, :250]], axis=1)
    g = 1.0 / np.linspace(0.95, 1.05, 5)
    a = multi_histogram(E, [E], 1.0 / betas, g, weights=w)[0]
    b = multi_histogram(rep, [rep], 1.0 / betas, g)[0]
    assert np.allclose(a, b, rtol=1e-9)


def test_e4_temperature_window_shrinks_with_energy_width():
    from Spin_Colour_Kernel.exp_perf import e4_temperatures
    Tc = 4.5
    small = e4_temperatures(Tc, sigma_E=50.0, ntemps=9, window=0.03)     # T/sigma_E = 9 %: capped at the window
    large = e4_temperatures(Tc, sigma_E=2000.0, ntemps=9, window=0.03)   # T/sigma_E = 0.225 %
    assert np.isclose(small[0], Tc * 0.97) and np.isclose(small[-1], Tc * 1.03)
    assert np.allclose(np.diff(large) / Tc, Tc / 2000.0)                 # neighbours one energy width apart
    assert np.isclose(np.mean(large), Tc) and len(large) == 9


def test_e4_pipeline_saves_series_and_declares_fields(tmp_path, monkeypatch):
    """Pipeline check with a synthetic sampler (the real one is covered by E1 and the H100 smoke):
    raw series are saved, every declared field is filled, temperatures are per size."""
    from Spin_Colour_Kernel import exp_perf as P

    def fake_series(s, L_box, temps, seeds, therm, meas, every, device):
        rng = np.random.default_rng(L_box)
        n = max(2, meas // every)
        E = rng.normal(-1.0 * L_box ** 2, 5.0 * L_box, size=(len(temps), seeds, n))
        m = np.abs(rng.normal(0.5, 0.1, size=(len(temps), seeds, n)))
        return E, m ** 2, m ** 4

    monkeypatch.setattr(P, "_series", fake_series)
    res = P.e4(tmp_path, device="cpu", sizes2d=(4, 6, 8), seeds=3, ntemps=5, meas=400, boots=5,
               only=["Ising square"], tag="E4-test", probe_cap=2000)
    r = res["rows"][0]
    assert r["reweighting"] == "multi-histogram" and r["sizes"] == [4, 6, 8]
    assert set(r["binder"]) == {"4-6", "6-8"}
    for L in (4, 6, 8):
        z = np.load(tmp_path / f"E4-test-Ising_square-L{L}.npz")
        assert z["E"].shape[:2] == (5, 3) and len(z["temps"]) == 5
        d = r["declared"][L]
        assert d["probe_sweeps"] == 2000 and d["tau_production"] > 0 and d["sigma_E"] > 0
        assert np.isclose(np.mean(r["temps"][L]), 2.26918531421, rtol=1e-6)


def test_sk_growth_supports_growth_below_and_saturation_above_crossover():
    from Spin_Colour_Kernel.exp_perf import sk_growth
    rng = np.random.default_rng(0)
    temps = [0.2, 0.4]
    # below: S_K proportional to L^2 / L^2 * L -> grows; above: saturated at 20 for every size
    means = {L: np.stack([L * (1 + 0.02 * rng.standard_normal(10)), 20 * (1 + 0.02 * rng.standard_normal(10))])
             for L in (24, 48, 96)}
    g = sk_growth(means, temps, 0.285, boots=300)
    assert g["verdict"] == "supported"
    assert g["below_all_grow"] and g["above_largest_pair_flat"]
    assert len(g["tests"]) == 4
    t = [x for x in g["tests"] if x["T"] == 0.2 and x["L1"] == 48][0]
    assert abs(t["log_ratio"] - np.log(2)) < 0.05 and t["low"] > 0


def test_sk_growth_rejects_growth_above_crossover_at_largest_pair():
    from Spin_Colour_Kernel.exp_perf import sk_growth
    rng = np.random.default_rng(1)
    means = {L: np.stack([L * (1 + 0.02 * rng.standard_normal(10))]) for L in (24, 48, 96)}
    g = sk_growth(means, [0.4], 0.285, boots=300)
    assert g["verdict"] == "not supported" and not g["above_largest_pair_flat"]


def test_e4_triangular_per_temperature_thinning_keeps_the_fast_temperatures_sampled(tmp_path, monkeypatch):
    import json
    from Spin_Colour_Kernel import exp_perf as P
    calls = []

    def fake(s, L, temps, seeds, therm, meas, every, device, seed):
        calls.append((meas, every))
        rng = np.random.default_rng(seed + meas)
        n = int(np.ceil(meas / every))
        # slow temperature (0.2): strongly correlated series; fast one (0.4): white noise
        slow = np.cumsum(rng.standard_normal((1, seeds, n)), axis=2) * 0.01 + 50.0
        fast = 20.0 + 0.1 * rng.standard_normal((1, seeds, n))
        return np.concatenate([slow, fast]), rng.standard_normal((len(temps), seeds, n))

    monkeypatch.setattr(P, "_sk_series", fake)
    P.e4_triangular_afm(tmp_path, device="cpu", sizes=(6,), temps=(0.2, 0.4), seeds=4, meas=200_000, boots=100,
                        probe_cap=4000, tag="slow", thin="slowest")
    P.e4_triangular_afm(tmp_path, device="cpu", sizes=(6,), temps=(0.2, 0.4), seeds=4, meas=200_000, boots=100,
                        probe_cap=4000, tag="perT", thin="per_T", seed_base=7)
    slow = json.loads((tmp_path / "slow.json").read_text())["rows"][0]
    perT = json.loads((tmp_path / "perT.json").read_text())
    r = perT["rows"][0]
    assert r["thin"] == "per_T" and perT["params"]["thin"] == "per_T" and perT["params"]["seed_base"] == 7
    assert r["every"] <= slow["every"] and r["samples_per_chain"] >= slow["samples_per_chain"]
    assert r["samples_per_chain"] <= 20_000
    assert "fss_slopes" in perT["growth"]


def test_e4_triangular_afm_declares_tau_and_saves_series(tmp_path, monkeypatch):
    import json
    from Spin_Colour_Kernel import exp_perf as P
    calls = []

    def fake(s, L, temps, seeds, therm, meas, every, device, seed):
        calls.append((L, therm, meas, every))
        rng = np.random.default_rng(seed + meas)
        n = int(np.ceil(meas / every))
        base = np.where(np.asarray(temps) < 0.285, float(L), 20.0)[:, None, None]
        return base * (1 + 0.01 * rng.standard_normal((len(temps), seeds, n))), rng.standard_normal((len(temps), seeds, n))

    monkeypatch.setattr(P, "_sk_series", fake)
    P.e4_triangular_afm(tmp_path, device="cpu", sizes=(6, 12), temps=(0.2, 0.4), seeds=4, meas=2000, boots=200)
    d = json.loads((tmp_path / "E4-tri.json").read_text())
    for r in d["rows"]:
        for k in ("tau", "probe_sweeps", "tau_resolved", "therm", "every", "tau_production_per_T"):
            assert k in r
        assert r["therm"] >= 1000
        assert (tmp_path / f"E4-tri-L{r['L']}.npz").exists()
    assert d["growth"]["verdict"] == "supported"
    assert any(c[1] == d["rows"][0]["therm"] for c in calls)   # production used the declared therm


def test_h4_cost_pairs_u4_grids_with_odd_neighbours():
    from Spin_Colour_Kernel.width_law_b import h4_cost
    rng = np.random.default_rng(0)
    rows = []
    for L1 in range(61, 73):
        base = 2.0 if L1 % 4 == 0 else 1.0                      # u = 4 costs twice
        rows.append({"factor": "layer-heis", "box": [L1, 31, 31],
                     "xla_ns_per_update": list(base * (1 + 0.02 * rng.standard_normal(10)))})
    rows.append({"factor": "layer", "box": [64, 31, 31], "xla_ns_per_update": [9.0] * 10})   # other sweep ignored
    r = h4_cost(rows, boots=300)
    assert [g["L1"] for g in r["grids"]] == [64, 68, 72]
    assert all(abs(g["log_ratio"] - np.log(2)) < 0.05 for g in r["grids"])
    assert r["supported"] and r["pooled"]["low"] > 0
    flat = [dict(x, xla_ns_per_update=[1.0 + 0.01 * k for k in range(10)]) for x in rows if x["factor"] == "layer-heis"]
    assert not h4_cost(flat, boots=300)["supported"]


def test_e5_conf_grids_follow_experiments_v04():
    import math
    from Spin_Colour_Kernel.width_law_b import grids_conf, rule, CONF_L1, MARGIN_H4
    for tC in (40960, 108 * 2048, 270336):           # T4, A100, H100
        g = grids_conf(tC)
        ab = [x for x in g if x[0] in ("below", "above")]
        assert len([x for x in ab if x[0] == "below"]) == 6 and len([x for x in ab if x[0] == "above"]) == 6
        for factor, B, n, b, box in ab:
            N = B * n * b * box[0] * box[1] * box[2]
            assert N % 4 == 0 and all(L % 2 for L in box)
            assert (N <= 0.8 * tC) if factor == "below" else (N >= 1.25 * tC)
            assert rule(N, tC) == (4 if factor == "above" else 1)
        heis = [x for x in g if x[0] == "layer-heis"]
        assert [x[4][0] for x in heis] == list(CONF_L1) and all(x[1:4] == (21, 3, 1) for x in heis)
        assert all(x[4][0] not in range(61, 73) for x in heis)          # not an E5c grid
        assert all(21 * 3 * x[4][0] * 961 >= tC for x in heis)          # every sweep grid above threshold
    assert abs(MARGIN_H4 - math.log(1.3)) < 1e-12


def test_e5_conf_verdict_reads_both_parts():
    from Spin_Colour_Kernel.width_law_b import conf_verdict
    rng = np.random.default_rng(1)
    tC = 1000
    rows = [{"factor": "below", "n": 1, "b": 1, "box": [5, 5, 5], "elements": 500, "logged_unroll": [1, 1]},
            {"factor": "above", "n": 1, "b": 1, "box": [7, 7, 7], "elements": 1372, "logged_unroll": [1, 4]}]
    for L1 in (75, 76, 77, 79, 80, 81, 83, 84, 85):
        f = 0.6 if L1 % 4 == 0 else 1.0                                   # u = 4 grids 1.7x faster
        rows.append({"factor": "layer-heis", "box": [L1, 31, 31], "sm_clock_mhz": [1980.0] * 10,
                     "xla_ns_per_update": list(f * (1 + 0.01 * rng.standard_normal(10)))})
    v = conf_verdict(rows, tC, boots=300)
    assert v["a"]["holds"] and v["b"]["holds"] and not v["b"]["inconclusive"] and v["b"]["sign"] == "u4 faster"
    bad = [dict(rows[0], logged_unroll=[4])] + rows[1:]                  # a factor above 1 below threshold
    assert not conf_verdict(bad, tC, boots=300)["a"]["holds"]
    small = [dict(r, xla_ns_per_update=[x * (0.9 if r["box"][0] % 4 == 0 else 1.0) / (0.6 if r["box"][0] % 4 == 0 else 1.0)
                                        for x in r["xla_ns_per_update"]]) if r["factor"] == "layer-heis" else r for r in rows]
    assert not conf_verdict(small, tC, boots=300)["b"]["holds"]          # |log 0.9| < log 1.3
    hot = [dict(r, sm_clock_mhz=[1500.0] * 10) if r["factor"] == "layer-heis" and r["box"][0] % 4 == 0 else r for r in rows]
    assert conf_verdict(hot, tC, boots=300)["b"]["inconclusive"]          # clocks differ by more than 5%


def test_e5_profile_parses_ncu_raw_csv():
    from Spin_Colour_Kernel.e5_profile import parse_raw_csv, derived
    head = ['"ID"', '"Kernel Name"', '"gpu__time_duration.sum"', '"launch__grid_size"', '"launch__block_size"',
            '"smsp__sass_inst_executed_op_global_ld.sum"', '"l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"',
            '"l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"', '"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"',
            '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"']
    text = "\n".join([",".join(head), ",".join(['""', '""', '"ns"'] + ['""'] * 7),
                      ",".join(['"0"', '"loop_select_fusion"', '"1,234"', '"100"', '"128"', '"1000"', '"512000"', '"1000"', '"16000"', '"7.5"']),
                      ",".join(['"1"', '"wrapped_x"', '"10"', '"1"', '"32"', '"0"', '"0"', '"0"', '"0"', '"0"'])])
    k = parse_raw_csv("==PROF== noise\n" + text)
    assert [x["kernel"] for x in k] == ["loop_select_fusion", "wrapped_x"]
    assert k[0]["gpu__time_duration.sum"] == 1234.0 and k[0]["stalls"] == {"long_scoreboard": 7.5}
    d = derived(k[0])
    assert d["threads"] == 12800 and d["bytes_per_thread_per_load"] == 16.0 and d["sectors_per_load_request"] == 16.0
    assert k[0]["units"]["gpu__time_duration.sum"] == "ns"
    assert derived(k[1])["bytes_per_thread_per_load"] is None


def test_e6_margin_is_log_1_05():
    import math
    from Spin_Colour_Kernel.exp_perf import MARGIN_H5
    assert abs(MARGIN_H5 - math.log(1.05)) < 1e-12


def test_nanodot_higher_shells_match_sequential_chain():
    from Spin_Colour_Kernel import system as sy
    from Spin_Colour_Kernel.sampler import Sampler
    base = sy.nanodot(Rd=2.5, L=3)
    s = sy.nanodot(Rd=2.5, L=3, shells=(1, 2, 4), J_shells={2: 0.3, 4: -0.1})
    assert s.J == {1: 1.0, 2: 0.3, 4: -0.1}
    assert s.colouring.n_colours > base.colouring.n_colours       # even shells break the checkerboard
    assert s.extra.keys() == base.extra.keys() or len(s.extra) == len(base.extra)   # DMI on the 4 in-plane bonds
    A, R = Sampler(s, 2, "cpu", 3), Sampler(s, 2, "cpu", 3)
    A.set_temperature(0.5)
    R.set_temperature(0.5)
    S1 = A.init_spins(seed=3)
    S2 = S1.clone()
    A.sweep(S1, 2)
    R.reference_sweep(S2, 2)
    assert int((S1 != S2).sum()) == 0


def test_nanodot_default_is_unchanged_first_shell():
    from Spin_Colour_Kernel import system as sy
    s = sy.nanodot(Rd=2.5, L=3)
    assert s.J == {1: 1.0} and s.colouring.n_colours == 2 and len(s.extra) == 4


def test_set_temperature_updates_beta_in_place():
    # a captured CUDA graph reads the beta buffer by address: changing T must not replace the buffer
    from Spin_Colour_Kernel import system as sy
    from Spin_Colour_Kernel.sampler import Sampler
    A = Sampler(sy.nanodot(Rd=2.5, L=3), 4, "cpu", 1)
    A.set_temperature(1.0)
    ptr = A.beta.data_ptr()
    A.set_temperature(torch.tensor([0.5, 1.0, 2.0, 4.0]))
    assert A.beta.data_ptr() == ptr
    assert torch.allclose(A.beta, torch.tensor([2.0, 1.0, 0.5, 0.25]))


def test_anneal_state_cools_and_writes_the_dataset_layout(tmp_path):
    import json
    from Spin_Colour_Kernel import anneal as AN
    rows = [[0.0, 0.0, 0.0, 0.2867, 0.0, 0.7691, 0.6535], [0.2, 0.0, 0.0, 0.2867, 0.0, 0.7691, 0.6535]]
    assert AN.active_shells(rows) == [1, 2]
    s = AN.build_state(rows[0], 2.5, 3, 90.0, [1])
    assert AN.check_exact(s, "cpu", sweeps=1) == 0
    temps = np.array([20.0, 0.1])
    r = AN.anneal_state(s, 3, temps, n_therm=20, n_meas=20, every=5, seed=1, device="cpu", log=lambda m: None)
    assert r["obs"].shape == (2, 8, 3) and r["mvec"].shape == (2, 3, 3) and r["snaps"].shape == (2, 7, 7)
    red = AN.reduce_obs(r["obs"], temps, s.n_active)
    assert np.all(np.isfinite(red["E"])) and np.all(red["Cv"] >= 0) and np.all(red["chi"] >= 0)
    assert red["E"][1] < red["E"][0] and red["M"][1] > red["M"][0]          # cooling orders the disk
    # the module entry point writes one .npz per state with the dataset layout
    cfg = dict(params=rows[:1], names=["uniform"], Rd=2.5, layers=3, angle=90.0, T_start=20.0, T_end=0.1,
               n_temps=2, sweeps=15, replicas=2, every=5, seed=42)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    AN.main(["--config", str(tmp_path / "run.json"), "--states", "0", "--out", str(tmp_path), "--device", "cpu"])
    z = np.load(tmp_path / "state_uniform.npz")
    assert z["M"].shape == (2, 2, 3) and z["temperatures"].shape == (2,) and z["params"].shape == (7,)


def test_observer_matches_sampler_observables():
    from Spin_Colour_Kernel import anneal as AN
    from Spin_Colour_Kernel.sampler import Sampler
    s = AN.build_state([0.1, 0.05, -0.05, 0.2867, 0.139, 0.7691, 0.6535], 2.5, 3, 60.0, [1, 2, 3, 4])
    A = Sampler(s, 3, "cpu", 2)
    S = A.init_spins(seed=2)
    E, Mv, Q = AN.Observer(A)(S)
    assert torch.allclose(E, A.energy(S), rtol=0, atol=1e-9)
    assert torch.allclose(Mv, A.magnetization(S)[:, 0, :] / s.n_active, atol=1e-12)
    ref = AN._topo(S[:, :, 0, s.L3 // 2].double(), AN.Observer(A).valid2d)
    assert torch.allclose(Q, ref)


CHUNKED = [sy.heisenberg("triangular", 24), sy.xy("square", 32),
           sy.System("Heisenberg sc compacted long rows", 3, cr.make("sc"), (645, 5, 5), (True,) * 3, {1: 1.0})]


@pytest.mark.parametrize("s", CHUNKED, ids=lambda s: s.name)
def test_rows_longer_than_max_block_are_chunked_exactly(s):
    """Rows longer than max_block run as several programs over grid axis 2 (no register spill at large L);
    the spins equal the one-program-per-row kernel and the sequential reference, compacted or not."""
    C = Sampler(s, 2, device="cpu", seed=5, max_block=16)
    U = Sampler(s, 2, device="cpu", seed=5, max_block=1024)
    F = Sampler(s, 2, device="cpu", seed=5, max_block=16, compact=False)
    R = Sampler(s, 2, device="cpu", seed=5, max_block=16)
    assert C.chunks > 1 and U.chunks == 1
    assert C.stride == 1 or C.cchunks > 1
    for X in (C, U, F, R):
        X.set_temperature(1.3)
    small = s.n_active <= 1024     # the sequential reference launches once per site: small systems only;
    S1 = C.init_spins(seed=6)      # the unchunked kernel equals the reference in the compacted-sweep test
    S2, S3, S4 = S1.clone(), S1.clone(), S1.clone()
    for _ in range(2 if small else 1):
        C.sweep(S1)
        U.sweep(S2)
        F.sweep(S3)
        if small:
            R.reference_sweep(S4)
            assert torch.equal(S1, S4)
        assert torch.equal(S1, S2)
        assert torch.equal(S1, S3)
    assert not torch.equal(S1, C.init_spins(seed=6))


def test_max_block_must_be_a_power_of_two():
    with pytest.raises(ValueError):
        Sampler(sy.xy("square", 8), 1, device="cpu", max_block=24)


def test_e5_profile_keeps_every_dram_and_device_metric():
    from Spin_Colour_Kernel.e5_profile import parse_raw_csv
    head = ['"ID"', '"Kernel Name"', '"dram__bytes.sum.per_second"', '"dram__bytes.sum.peak_sustained"',
            '"device__attribute_display_name"']
    text = "\n".join([",".join(head), ",".join(['""', '""', '"Gbyte/s"', '"byte/cycle"', '""']),
                      ",".join(['"0"', '"kernel"', '"1,500.5"', '"2,048"', '"NVIDIA H100 80GB HBM3"'])])
    k = parse_raw_csv(text)[0]
    assert k["dram__bytes.sum.per_second"] == 1500.5 and k["dram__bytes.sum.peak_sustained"] == 2048.0
    assert k["device__attribute_display_name"] == "NVIDIA H100 80GB HBM3"
    assert k["units"]["dram__bytes.sum.per_second"] == "Gbyte/s"


def test_e5_profile_roofline_targets_exceed_the_h100_l2():
    from Spin_Colour_Kernel.e5_profile import GEN
    for spec in ("heis-sc-64", "heis-fcc-32"):
        kind, crystal, L, B, packed = GEN[spec]
        basis = {"sc": 1, "fcc": 4}[crystal]
        assert kind == "heisenberg" and L ** 3 * basis * B * 3 * 4 > 50 * 2 ** 20


def test_periodic_pair_grids_are_even_and_admit_a_periodic_colouring():
    from Spin_Colour_Kernel import crystals as cr
    from Spin_Colour_Kernel import system as sy
    from Spin_Colour_Kernel.width_law_b import grids_pp
    g = grids_pp(threshold=270336)          # H100 resident-thread count (132 SMs x 2048)
    assert {x[0] for x in g} >= {"below", "above", "layer-heis", "layer"}
    for factor, B, n, b, box in g:
        assert all(L % 2 == 0 for L in box), box
    crystal_for_b = {1: "sc", 2: "bcc", 4: "fcc"}
    for factor, B, n, b, box in {(x[0], x[1], x[2], x[3], x[4]) for x in g if x[0] in ("below", "above")}:
        s = sy.System("pp", n, cr.make(crystal_for_b[b]), box, (True,) * 3, {1: 1.0})
        assert s.colouring.n_colours <= 4
