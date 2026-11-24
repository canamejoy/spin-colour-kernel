"""describe() and check(): the vocabulary and the validation an agent reads before writing or launching a run."""

import pytest

from Spin_Colour_Kernel import agent, runfile


def test_describe_lists_every_choice_and_the_shipped_defaults():
    d = agent.describe()
    assert set(d["models"]) == {"ising", "xy", "heisenberg"}
    assert set(d["crystals"]) == {"square", "triangular", "honeycomb", "kagome", "sc", "bcc", "fcc"}
    assert d["coordinates"]["cylindrical"] == ["rho", "phi", "z"]
    assert set(d["state_parameters"]) == {"J", "D", "K1", "Ks", "K2", "H", "H_theta", "H_phi"}
    assert d["defaults"] == runfile.defaults() and d["formats"] == ["ipynb", "py"]


def test_check_reports_sites_colouring_and_budget_of_the_default_run():
    r = agent.check({})
    assert r["ok"] and r["errors"] == []
    assert r["sites"] == dict(box=1728, active=1728, surface=0, bulk=1728)
    assert [s["colours"] for s in r["states"]] == [2, 2]
    assert r["temperatures"] == dict(count=19, first=2.0, last=0.2)
    assert r["budget"]["chains"] == 64 and r["budget"]["spin_updates"] == 64 * 19 * 1728 * 3000


def test_check_counts_the_surface_of_a_disc():
    r = agent.check(dict(box=[16, 16, 3], periodic=[False, False, False], coords="cylindrical", shape="rho <= 6",
                         states=[dict(name="d", J={"1": 1.0}, K1=0.1, Ks=0.2)]))
    assert r["ok"] and 0 < r["sites"]["surface"] < r["sites"]["active"] < r["sites"]["box"]


@pytest.mark.parametrize("cfg,needle", [
    (dict(boxx=[4]), "unknown setting 'boxx'"),
    (dict(model="potts"), "unknown model 'potts'"),
    (dict(crystal="sc", box=[4, 4]), "box must be a list of 3 values"),
    (dict(T_min=3.0), "need T_max >= T_min > 0"),
    (dict(coords="cartesian", shape="x > 1000"), "contains no site"),
    (dict(coords="cartesian", shape="__import__('os')"), "calls something other than"),
    (dict(coords="cartesian", shape="rho < 2"), "name 'rho' is not defined; the coordinates are x, y, z"),
    (dict(model="xy", crystal="square", box=[8, 8], periodic=[True, True], states=[dict(name="a", D=1.0)]),
     "need the Heisenberg model"),
    (dict(replicas=0), "replicas must be at least 1"),
])
def test_check_returns_errors_instead_of_raising(cfg, needle):
    r = agent.check(cfg)
    assert not r["ok"] and any(needle in e for e in r["errors"]), r["errors"]
