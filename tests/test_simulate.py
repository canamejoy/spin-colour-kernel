"""The general simulation module behind notebooks/spin_simulation.ipynb (CPU, Triton interpreter)."""
import json

import numpy as np
import pytest

from Spin_Colour_Kernel import simulate as SIM
from Spin_Colour_Kernel import system as sy


def test_an_empty_shape_keeps_every_site_active():
    pos, mask = SIM.active_sites("sc", (4, 4, 4), "", "")
    assert mask.all() and mask.shape == (1, 4, 4, 4) and pos.shape == (1, 4, 4, 4, 3)


def test_a_sphere_in_spherical_coordinates_selects_the_sites_inside_it():
    pos, mask = SIM.active_sites("sc", (9, 9, 9), "spherical", "r <= 3")
    r = np.linalg.norm(pos, axis=-1)
    assert mask.any() and not mask.all()
    assert np.all(r[mask] <= 3 + 1e-9) and np.all(r[~mask] > 3)


def test_a_cylinder_in_cylindrical_coordinates_bounds_rho_and_z():
    pos, mask = SIM.active_sites("sc", (11, 11, 6), "cylindrical", "(rho <= 4) & (abs(z) <= 1.5)")
    rho = np.hypot(pos[..., 0], pos[..., 1])
    assert np.all(rho[mask] <= 4 + 1e-9) and np.all(np.abs(pos[..., 2][mask]) <= 1.5 + 1e-9)


def test_an_unknown_coordinate_system_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="cartesian"):
        SIM.active_sites("sc", (4, 4, 4), "toroidal", "r < 1")


def test_a_shape_without_its_coordinate_system_is_refused():
    with pytest.raises(ValueError, match="coordinate"):
        SIM.active_sites("sc", (4, 4, 4), "", "r < 1")


def test_the_dmi_term_on_sc_matches_the_nanodot_of_the_paper():
    theta = {"J": {1: 1.0}, "D": 0.88, "K1": 0.177, "Ks": 0.139, "K2": 0.034, "H": 0.483,
             "H_theta": 0.0, "H_phi": 0.0}
    nd = sy.nanodot(Rd=3.3, L=3)
    s = SIM.build_system("heisenberg", "sc", nd.box, (False, False, False), theta,
                         mask=nd.site_mask())
    for i in range(len(s.crystal.bonds)):
        assert np.allclose(s.bond_matrix(i), nd.bond_matrix(i))
    assert np.allclose(s.cubic_k1, nd.cubic_k1) and s.cubic_k2 == nd.cubic_k2
    assert np.allclose(s.h, (0.0, 0.0, 0.483))


def test_terms_a_model_cannot_carry_are_refused():
    with pytest.raises(ValueError, match="Heisenberg"):
        SIM.build_system("ising", "square", (8, 8), (True, True), {"J": {1: 1.0}, "D": 0.5})


def test_one_state_returns_every_declared_array_with_its_shape():
    s = SIM.build_system("xy", "square", (6, 6), (True, True), {"J": {1: 1.0}, "H": 0.1})
    temps = SIM.temperature_ladder(2.0, 1.0, 0.5)
    pos, mask = SIM.active_sites("square", (6, 6), "", "")
    r = SIM.simulate_state(s, replicas=3, temps=temps, n_therm=4, n_meas=4, every=2, config_every=2,
                           seed=1, device="cpu", kB=1.0, log=lambda m: None)
    nT = len(temps)
    assert list(temps) == [2.0, 1.5, 1.0]
    assert r["M"].shape == (3, nT, 3) and r["E"].shape == (3, nT) and r["E2"].shape == (3, nT)
    assert r["Mabs"].shape == (3, nT) and r["M2"].shape == (3, nT)
    assert r["configs"].shape == (3, 2, 36, 3) and list(r["config_temperatures"]) == [2.0, 1.0]
    norms = np.linalg.norm(r["configs"], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.all(r["configs"][..., 2] == 0)                  # XY spins carry no third component
    assert np.all(r["Mabs"] <= 1 + 1e-6) and np.all(r["E2"] >= r["E"] ** 2 - 1e-6)


def test_ising_with_a_full_word_of_replicas_reads_unpacked_spins():
    s = SIM.build_system("ising", "square", (4, 4), (True, True), {"J": {1: 1.0}})
    r = SIM.simulate_state(s, replicas=32, temps=[3.0], n_therm=2, n_meas=2, every=1, config_every=1,
                           seed=1, device="cpu", kB=1.0, log=lambda m: None)
    c = r["configs"]
    assert c.shape == (32, 1, 16, 3) and set(np.unique(c[..., 0])) <= {-1.0, 1.0}


def test_the_run_writes_the_combined_arrays_for_every_state(tmp_path):
    cfg = SIM.default_config()
    cfg.update(model="ising", crystal="square", box=[6, 6], periodic=[True, True],
               states=[{"name": "a", "J": {"1": 1.0}}, {"name": "b", "J": {"1": 1.0}, "H": 0.2}],
               replicas=2, T_max=3.0, T_min=2.0, T_step=0.5, thermalization=3, measurement=3,
               measure_every=1, units="reduced", seed=1)
    out = tmp_path / "out"
    SIM.prepare(cfg, out)
    SIM.run(json.loads((out / "run.json").read_text()), [0, 1], out, device="cpu", log=lambda m: None)
    res = SIM.collect(out)
    assert res["magnetization"].shape == (2, 2, 3, 3)
    cf = np.load(out / "configurations.npy", mmap_mode="r")
    assert cf.shape == (2, 2, 3, 36, 3)
    assert np.all(np.abs(cf[..., 0]) == 1)
    assert res["energy"].shape == (2, 2, 3) and res["Cv"].shape == (2, 3) and res["chi"].shape == (2, 3)
    assert np.load(out / "sites.npy").shape == (36, 3)


def test_the_size_estimate_counts_the_configuration_array():
    gb = SIM.size_estimate_gb(n_states=2, replicas=128, n_saved=200, n_sites=5000)
    assert abs(gb - 2 * 128 * 200 * 5000 * 3 * 4 / 1e9) < 1e-9


def test_replicas_are_split_equally_with_every_replica_covered_once():
    assert SIM.replica_blocks(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert SIM.replica_blocks(32, 2) == [(0, 16), (16, 32)]
    assert SIM.replica_blocks(3, 8) == [(0, 1), (1, 2), (2, 3)]          # never an empty block


def test_replica_blocks_run_apart_combine_into_the_full_replica_axis(tmp_path):
    cfg = SIM.default_config()
    cfg.update(model="ising", crystal="square", box=[6, 6], periodic=[True, True],
               states=[{"name": "a", "J": {"1": 1.0}}, {"name": "b", "J": {"1": 1.0}, "H": 0.2}],
               replicas=5, T_max=3.0, T_min=2.0, T_step=0.5, thermalization=3, measurement=3,
               measure_every=1, units="reduced", seed=1)
    out = tmp_path / "out"
    SIM.prepare(cfg, out)
    run = json.loads((out / "run.json").read_text())
    blocks = SIM.replica_blocks(5, 2)
    for part, rng in enumerate(blocks):                                 # as two GPUs would, one block each
        SIM.run(run, [0, 1], out, device="cpu", log=lambda m: None, replicas=rng, part=part)
    res = SIM.collect(out)
    assert res["magnetization"].shape == (2, 5, 3, 3) and res["energy"].shape == (2, 5, 3)
    cf = np.load(out / "configurations.npy", mmap_mode="r")
    assert np.all(np.abs(cf[..., 0]) == 1)                               # every replica slot was written
    # the two blocks are different chains, not copies of one another
    assert not np.array_equal(cf[0, 0], cf[0, 3])


def test_collect_refuses_a_replica_axis_with_a_missing_block(tmp_path):
    cfg = SIM.default_config()
    cfg.update(model="ising", crystal="square", box=[4, 4], periodic=[True, True],
               states=[{"name": "a", "J": {"1": 1.0}}], replicas=4, T_max=2.0, T_min=2.0, T_step=0.5,
               thermalization=1, measurement=1, measure_every=1, units="reduced", seed=1)
    out = tmp_path / "out"
    SIM.prepare(cfg, out)
    run = json.loads((out / "run.json").read_text())
    SIM.run(run, [0], out, device="cpu", log=lambda m: None, replicas=(0, 2), part=0)
    with pytest.raises(ValueError, match="replicas"):
        SIM.collect(out)


def test_the_graph_safe_observer_matches_the_sampler_energy_and_magnetization():
    theta = {"J": {1: 1.0}, "D": 0.88, "K1": 0.177, "Ks": 0.139, "K2": 0.034, "H": 0.483}
    nd = sy.nanodot(Rd=3.3, L=3)
    for s in (SIM.build_system("heisenberg", "sc", nd.box, (False, False, False), theta, mask=nd.site_mask()),
              SIM.build_system("xy", "triangular", (6, 6), (True, True), {"J": {1: 1.0, 2: -0.3}, "H": 0.1}),
              SIM.build_system("ising", "square", (4, 4), (True, True), {"J": {1: 1.0}, "H": 0.2})):
        from Spin_Colour_Kernel.sampler import Sampler
        A = Sampler(s, 3, "cpu", 1)
        S = A.init_spins(seed=2)
        E, Mv = SIM._Observer(A)(S)
        assert np.allclose(E.numpy(), A.energy(S).numpy(), atol=1e-9)
        assert np.allclose(Mv.numpy(), (A.magnetization(S).sum(dim=1) / s.n_active).numpy(), atol=1e-12)


@pytest.mark.parametrize("shape", ["().__class__.__bases__[0].__subclasses__()", "__import__('os').system('true')",
                                   "(lambda: 1)()", "x.real <= 2", "'a' < 'b'", "open('f')", "abs(x, key=1) < 2"])
def test_a_shape_that_is_not_a_condition_on_the_coordinates_is_refused_before_it_runs(shape):
    with pytest.raises(ValueError):
        SIM.active_sites("sc", [4, 4, 4], "cartesian", shape)


def test_a_shape_with_and_or_not_is_refused_with_the_operators_to_use():
    with pytest.raises(ValueError, match=r"use & \(and\), \| \(or\), ~ \(not\)"):
        SIM.active_sites("sc", [4, 4, 4], "cartesian", "(x < 1) and (y < 1)")


@pytest.mark.parametrize("coords,shape", [("cylindrical", "(rho <= 3) & (abs(z) <= 1)"), ("spherical", "r <= 2.5"),
                                          ("cartesian", "~((abs(x) < 1) | (abs(y) < 1)) ^ (z > 0)"),
                                          ("cylindrical", "(rho <= 3) & (rho >= 1.5)"),
                                          ("spherical", "(cos(theta) > 0) & (r ** 2 <= 2 * pi)")])
def test_a_condition_on_the_coordinates_still_selects_sites(coords, shape):
    pos, mask = SIM.active_sites("sc", [6, 6, 6], coords, shape)
    assert 0 < mask.sum() < mask.size
