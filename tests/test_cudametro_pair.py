"""CPU-only checks of the CUDA-METRO pairing helper: temperature mapping, material and config files, the rate
arithmetic, the P choice, the worker-output parser and the JSON row assembly with fake timings."""

import json
import math
import re

import numpy as np
import pytest

from Spin_Colour_Kernel import cudametro_pair as C


def _read_2dmat(path):
    """CUDA-METRO's own parser (Material_Reader.read_2dmat at the pinned commit), restated for the test."""
    names, params = [], []
    for line in path.read_text().splitlines():
        sub = re.split(r"[,|+]", line)
        names.append(sub[0])
        params += [np.float32(x) for x in sub[1:]]
    return names, np.array(params, dtype=np.float32)


# ------------------------------------------------------------------ temperature
@pytest.mark.parametrize("T", [0.5, 1.0, 1.2, 2.0])
def test_temperature_mapping(T):
    # construct.py: beta = 1 / (T_K * 8.6173e-2) in 1/meV; with J1 = J_MEV meV the kernel's dE * beta is dE / T
    assert C.kelvin(T) == pytest.approx(T * C.J_MEV / C.KB_MEV, rel=1e-12)
    b = C.beta(T)
    assert b.dtype == np.float32 and b.shape == (1,)
    assert float(b[0]) == pytest.approx(1.0 / (T * C.J_MEV), rel=1e-6)


# ------------------------------------------------------------------ material and config
@pytest.mark.parametrize("lattice,nbs", [("square", [4, 4, 4, 8]), ("triangular", [6, 6, 6, 12])])
def test_material_file_sets_J1_only(tmp_path, lattice, nbs):
    path = C.write_material(tmp_path, lattice)
    assert path.name == f"TC_{C.MATERIAL}.csv"          # construct.py reads Input_Folder + "TC_" + Material + ".csv"
    names, p = _read_2dmat(path)
    assert names == [C.MATERIAL]
    assert len(p) == 26
    assert p[0] == 1.0                                   # spin magnitude
    assert p[1] == pytest.approx(C.J_MEV)                # J1
    assert np.all(p[2:20] == 0.0)                        # J2-J4, anisotropic exchange K1-K4, single-ion A
    assert list(p[20:24].astype(int)) == nbs             # neighbour-shell signature
    assert p[25] == 0.0                                  # DMI


def test_unsupported_lattice_rejected(tmp_path):
    with pytest.raises(ValueError, match="honeycomb"):
        C.write_material(tmp_path, "honeycomb")
    with pytest.raises(ValueError):
        C.config_for("kagome", 64, 128, 10, 1.0)


def test_config_mirrors_cudametro_keys():
    c = C.config_for("square", 256, 1024, 19531, 1.0)
    assert c["SIZE"] == 256 and c["Blocks"] == 1024 and c["Threads"] == 2
    assert c["stability_runs"] == 19531 and c["stability_wrap"] == 1
    assert c["Material"] == C.MATERIAL
    assert c["DMI_Flag"] == 0 and c["TC_Flag"] == 1 and c["Static_T_Flag"] == 1
    assert float(c["B"]) == 0.0
    assert float(c["Temps"][0]) == pytest.approx(C.kelvin(1.0))
    assert c["Lattice"] == "square"
    json.dumps(c)                                        # serialisable as CUDA-METRO's JSON config


# ------------------------------------------------------------------ rates and P choice
def test_rate_conversion():
    # P sites per round, rounds rounds, seconds of kernel time -> attempts per ns
    assert C.metro_rate(1024, 1000, 0.5) == pytest.approx(1024 * 1000 / 0.5e9)
    assert math.isnan(C.metro_rate(1024, 1000, 0.0))
    assert math.isnan(C.metro_rate(1024, 1000, -1.0))


def test_rounds_for_attempts():
    assert C.rounds_for(2e7, 1024) == 2e7 // 1024
    assert C.rounds_for(10, 1024) == 1


def test_p_grid_and_ceiling():
    assert C.p_grid(64 * 64) == [64, 128, 256, 512, 1024, 2048, 4096]
    assert max(C.p_grid(2048 ** 2)) == 16384
    N = 256 * 256
    acc = math.floor(0.01 * N / 0.0276)                 # analyze_bench.py: P <= (1 - A) N / 0.0276, A = 0.99
    assert C.accuracy_ceiling(N) == acc
    assert C.usable_ceiling(N, hw=3456) == 3456          # hardware binds
    assert C.usable_ceiling(64 * 64, hw=3456) == math.floor(0.01 * 4096 / 0.0276)   # accuracy binds
    assert C.usable_ceiling(N, hw=None) == acc


def test_pick_fastest_ignores_nan():
    assert C.pick_fastest({64: 0.1, 1024: 0.9, 4096: float("nan"), 2048: 0.5}) == 1024
    with pytest.raises(ValueError):
        C.pick_fastest({64: float("nan")})


# ------------------------------------------------------------------ worker output
def test_parse_worker_output():
    rec = {"Blocks": 1024, "rounds": 10, "kernel_s": 0.01}
    text = "compiling...\n" + C.RESULT_TAG + json.dumps(rec) + "\nnoise\n" + C.RESULT_TAG + json.dumps(rec) + "\n"
    assert C.parse_worker_output(text) == [rec, rec]
    with pytest.raises(RuntimeError):
        C.parse_worker_output("Traceback: boom")


# ------------------------------------------------------------------ row assembly
def test_assemble_row_with_fake_timings():
    metro = [0.10, 0.11, 0.09, 0.10]
    ours = {1: [1.0, 1.1, 0.9, 1.0], "1d": [0.5, 0.55, 0.45, 0.5]}
    row = C.assemble_row("square", 256, 65536, 1024, ["fastest", "ceiling"], metro, ours, (1, "1d"),
                         runs=[{"kernel_s": 0.2}] * 4)
    assert row["system"] == "Heisenberg square"
    assert row["L"] == 256 and row["active_sites"] == 65536 and row["cudametro_sites"] == 65536
    assert row["blocks_P"] == 1024 and row["p_choice"] == ["fastest", "ceiling"]
    assert row["P_over_N"] == pytest.approx(1024 / 65536)
    assert row["predicted_error"] == pytest.approx(0.0276 * 1024 / 65536)
    assert row["metro_flips_per_ns"] == metro
    assert row["ours_flips_per_ns"] == {"1": ours[1], "1d": ours["1d"]}
    for B in (1, "1d"):
        lr = row[f"log_ratio_B{B}"]
        expect = float(np.mean(np.log(np.asarray(ours[B]) / np.asarray(metro))))
        assert lr["mean"] == pytest.approx(expect)
        assert lr["low"] <= lr["mean"] <= lr["high"]
    assert row["log_ratio_B1"]["mean"] == pytest.approx(math.log(10), abs=0.05)


def test_assemble_row_length_mismatch():
    with pytest.raises(ValueError):
        C.assemble_row("square", 64, 4096, 64, ["fastest"], [0.1, 0.2], {1: [1.0]}, (1,))


def test_note_states_exact_vs_approximate():
    n = C.NOTE.lower()
    assert "exact" in n and "0.0276" in n and "approximate" in n
    assert C.CUDAMETRO_COMMIT == "cbc9737f18ed7827efea95f3791ff7e99e24dc87"
