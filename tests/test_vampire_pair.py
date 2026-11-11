"""CPU-only checks of the VAMPIRE pairing helper: input files, the rate arithmetic and the log parser."""

import math

import pytest

from Spin_Colour_Kernel import vampire_pair as V


def _kv(path):
    """key -> (value, unit) of a VAMPIRE input/material file, comments dropped (format: key = value !unit)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        unit = ""
        if "!" in line:
            line, unit = (p.strip() for p in line.split("!", 1))
        key, _, val = line.partition("=")
        out[key.strip()] = (val.strip(), unit)
    return out


@pytest.mark.parametrize("crystal", ["sc", "bcc", "fcc"])
def test_write_inputs_keywords(tmp_path, crystal):
    info = V.write_inputs(tmp_path, crystal, cells=32, T=1.2, sweeps=500, seed=7)
    inp = _kv(tmp_path / "input")
    mat = _kv(tmp_path / V.MAT_FILE)

    assert inp["create:crystal-structure"][0] == crystal
    for ax in "xyz":
        assert f"create:periodic-boundaries-{ax}" in inp
    a = float(inp["dimensions:unit-cell-size"][0])
    assert inp["dimensions:unit-cell-size"][1] == "A"
    for ax in "xyz":
        size, unit = inp[f"dimensions:system-size-{ax}"]
        assert unit == "A"
        assert math.ceil(float(size) / a) == 32           # VAMPIRE rounds size / a up (create.cpp)
    assert inp["material:file"][0] == V.MAT_FILE
    assert inp["sim:program"][0] == "time-series"
    assert inp["sim:integrator"][0] == "monte-carlo"
    assert inp["montecarlo:algorithm"][0] == "uniform"
    assert inp["gpu:acceleration"][0] == "on"
    assert inp["sim:total-time-steps"][0] == "500"
    assert inp["sim:time-steps-increment"][0] == "500"
    assert inp["sim:equilibration-time-steps"][0] == "0"
    assert inp["sim:integrator-random-seed"][0] == "7"
    T_K = 1.2 * V.J_LINK / V.KB
    assert float(inp["sim:temperature"][0]) == pytest.approx(T_K, rel=1e-9)
    assert float(inp["sim:equilibration-temperature"][0]) == pytest.approx(T_K, rel=1e-9)
    assert info["temperature_K"] == pytest.approx(T_K, rel=1e-9)
    assert info["n_atoms_expected"] == 32 ** 3 * {"sc": 1, "bcc": 2, "fcc": 4}[crystal]
    assert not any(k.startswith("output:") or k.startswith("screen:") for k in inp)   # minimal output

    assert mat["material:num-materials"][0] == "1"
    assert float(mat["material[1]:exchange-matrix[1]"][0]) == pytest.approx(V.J_LINK)
    assert float(mat["material[1]:second-order-uniaxial-anisotropy-constant"][0]) == 0.0
    assert "material[1]:atomic-spin-moment" in mat


def test_write_inputs_cpu_mode_and_bad_crystal(tmp_path):
    V.write_inputs(tmp_path, "sc", cells=4, T=1.2, sweeps=3, seed=1, gpu=False)
    assert _kv(tmp_path / "input")["gpu:acceleration"][0] == "off"
    with pytest.raises(ValueError):
        V.write_inputs(tmp_path, "hcp", cells=4, T=1.2, sweeps=3, seed=1)


def test_temperature_mapping_is_reduced_units():
    # J / (kB T_K) must equal 1 / T for every reduced T
    for T in (0.5, 1.2, 3.0):
        T_K = V.kelvin(T)
        assert V.J_LINK / (V.KB * T_K) == pytest.approx(1.0 / T, rel=1e-12)


def test_flips_per_ns_arithmetic():
    assert V.flips_per_ns(131072, 1000, 2.0) == pytest.approx(131072 * 1000 / 2e9)
    # baseline-subtracted: n (S - S0) / (t_S - t_S0)
    assert V.flips_per_ns(32768, 1000 - 10, 1.5 - 0.5) == pytest.approx(32768 * 990 / 1e9)
    assert math.isnan(V.flips_per_ns(10, 10, 0.0))
    assert math.isnan(V.flips_per_ns(10, 10, -1.0))


# Synthetic stdout built from the print statements at VAMPIRE 525bc27: create.cpp:196-199 (atom counts),
# cuda/initialize.cu:52-56 (precision), cuda/monte_carlo.cu colour_split (colours),
# simulate/sim.cpp:542-547 (run time, Monte Carlo statistics).
LOG = """\
Total number of atoms (all CPUs): 131072
Number of atoms generated: 131072
CUDA has been enabled in double precision mode
Starting graph colouring
No more uncoloured atoms. Fraction coloured = 131072 / 131072
Colour: 0, natoms = 32768
Colour: 1, natoms = 32768
Colour: 2, natoms = 32768
Colour: 3, natoms = 32768
Trying a step...
Done
Simulation run time [s]: 12.5
Monte Carlo statistics:
\tTotal moves: 131203072
\t61.2% Accepted
\t38.8% Rejected
"""


def test_parse_log_gpu():
    f = V.parse_log(LOG)
    assert f["n_atoms"] == 131072
    assert f["gpu"] is True
    assert f["precision"] == "double"
    assert f["colours"] == [32768] * 4
    assert f["n_colours"] == 4
    assert f["sim_seconds"] == pytest.approx(12.5)
    assert f["total_moves"] == 131203072
    assert f["accepted_pct"] == pytest.approx(61.2)


def test_parse_log_cpu_without_gpu_lines():
    f = V.parse_log("Number of atoms generated: 64\nSimulation run time [s]: 0.25\n")
    assert f["n_atoms"] == 64 and f["gpu"] is False and f["n_colours"] == 0
    assert f["sim_seconds"] == pytest.approx(0.25)
    assert f["total_moves"] is None
