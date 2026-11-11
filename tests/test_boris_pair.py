"""CPU-only checks of the BORIS pairing helper: temperature mapping, generated console script, build recipe,
result parsing and row assembly. Nothing here builds or runs BORIS or touches a GPU."""

import json
import math
import sys
import types

import numpy as np
import pytest

from Spin_Colour_Kernel import boris_pair as B
from Spin_Colour_Kernel import vampire_pair as V


# ------------------------------------------------------------------ temperature
def test_temperature_mapping_is_reduced_units_with_boris_kb():
    assert B.KB == 1.3806488e-23                     # BorisLib/Funcs_Math_base.h BOLTZMANN
    assert B.J_LINK == V.J_LINK
    for T in (0.5, 1.2, 3.0):
        assert B.J_LINK / (B.KB * B.kelvin(T)) == pytest.approx(1.0 / T, rel=1e-12)


def test_arch_matches_boris_device_check():
    # Simulation.cpp selects a device only when major * 100 == __CUDA_ARCH__ = arch * 10
    assert B.arch_for_capability((8, 0)) == "80"
    assert B.arch_for_capability((8, 6)) == "80"
    assert B.arch_for_capability((9, 0)) == "90"
    assert B.arch_for_capability((10, 0)) == "100"


# ------------------------------------------------------------------ console commands and script
@pytest.mark.parametrize("cells", [32, 48, 64])
def test_setup_commands_sc(cells):
    cmds = B.setup_commands("sc", cells, T=1.2, seed=11)
    L = cells * B.LATTICE_A
    assert cmds[0] == f"setameshcubic {B.MESH} 0 0 0 {L!r} {L!r} {L!r}"
    assert f"cellsize {B.MESH} {B.LATTICE_A!r}" in cmds
    assert round(L / B.LATTICE_A) == cells                     # Atom_Mesh_Cubic.cpp: n = round(meshRect / h)
    for ax in "xyz":
        assert f"pbc {B.MESH} {ax} 1" in cmds
    assert f"delmodule {B.MESH} dipoledipole" in cmds
    assert f"delmodule {B.MESH} Zeeman" in cmds
    assert not any(c.startswith("addmodule") for c in cmds)    # exchange only (a default module)
    assert f"setparam {B.MESH} J {B.J_LINK!r}" in cmds
    assert f"temperature {B.MESH} {B.kelvin(1.2)!r}" in cmds
    assert f"prngseed {B.MESH} 11" in cmds
    assert "setstage MonteCarlo" in cmds
    assert f"mcserial {B.MESH} 0" in cmds                      # parallel red/black (the only CUDA path)
    assert f"mcconstrain {B.MESH} 0" in cmds                   # classical, unconstrained Metropolis
    assert "mccomputefields 0" in cmds
    assert "iterupdate 0" in cmds
    assert cmds[-1] == "cuda 1"
    assert B.n_spins("sc", cells) == cells ** 3


@pytest.mark.parametrize("crystal", ["bcc", "fcc", "hcp"])
def test_unsupported_crystals_rejected(crystal):
    assert B.SUPPORTED_CRYSTALS == ("sc",)
    with pytest.raises(ValueError):
        B.setup_commands(crystal, 32, T=1.2, seed=1)
    with pytest.raises(ValueError):
        B.write_script(None, crystal, 32, 1.2, 1, warm=1, baseline=2, full=3)


class _FakeBPython:
    """Stands in for BORIS's embedded module (Boris/PythonScripting.h): records messages, returns tab-led fields."""

    def __init__(self, bench_ms):
        self.sent, self.runs, self.bench = [], 0, list(bench_ms)

    def BPython_Message(self, s):
        self.sent.append(s)
        if s == "*benchtime":
            return "\t%d" % self.bench[self.runs - 1]
        if s == "*cuda":
            return "\t1"
        return "\t"

    def BPython_Message_Run(self):
        self.runs += 1
        return ""


def test_script_runs_against_fake_bpython(tmp_path, monkeypatch, capsys):
    script = B.write_script(tmp_path, "sc", 32, 1.2, 5, warm=7, baseline=10, full=110)
    text = script.read_text()
    # BORIS rewrites `.Atomistic(`-style calls and indents the script inside a method (Commands_PythonScripting.cpp)
    for h in ("Ferromagnet", "AntiFerromagnet", "Dipole", "Conductor", "Insulator", "Atomistic", "Material"):
        assert f".{h}(" not in text
    assert '"""' not in text and "'''" not in text
    fake = _FakeBPython([900, 40, 1040])
    monkeypatch.setitem(sys.modules, "BPython", fake)
    exec(compile(text, str(script), "exec"), {"__name__": "__main__"})
    res = json.loads((tmp_path / B.RESULT_FILE).read_text())
    assert fake.runs == 3
    sent = [s[1:] for s in fake.sent]
    assert sent[:len(B.setup_commands("sc", 32, 1.2, 5))] == B.setup_commands("sc", 32, 1.2, 5)
    assert sent.count("reset") == 3
    assert "editstagestop 0 iter 7" in sent and "editstagestop 0 iter 10" in sent and "editstagestop 0 iter 110" in sent
    assert [r["name"] for r in res["runs"]] == ["warm", "baseline", "full"]
    assert [r["iters"] for r in res["runs"]] == [7, 10, 110]
    assert [B.ret_number(r["benchtime"]) for r in res["runs"]] == [900, 40, 1040]
    assert "error" not in res
    out = capsys.readouterr().out
    assert B.MARKER in out


# ------------------------------------------------------------------ parsing and rates
SAMPLE = (
    "Started Python script : /w/sc32/rep0/pair.py\n"
    "Initializing modules...\n"
    + B.MARKER + " " + json.dumps({
        "params": {"crystal": "sc", "cells": 32},
        "cuda": "\t1",
        "runs": [{"name": "warm", "iters": 10, "wall_s": 1.1, "benchtime": "\t95"},
                 {"name": "baseline", "iters": 10, "wall_s": 0.2, "benchtime": "\t20"},
                 {"name": "full", "iters": 1010, "wall_s": 2.3, "benchtime": "\t2020"}],
        "mcparams": "\t45\t0.5"}) + "\n"
    "Finished Python script : /w/sc32/rep0/pair.py\n")


def test_ret_number():
    assert B.ret_number("\t2020") == 2020.0
    assert B.ret_number("\t45\t0.5") == 45.0
    assert B.ret_number("") is None and B.ret_number("\t") is None and B.ret_number(None) is None
    assert B.ret_number(7) == 7.0


def test_parse_log_extracts_marker_and_rates():
    res = B.parse_log(SAMPLE)
    assert res["runs"][2]["iters"] == 1010
    r = B.rates(res, n_atoms=32 ** 3)
    assert r["cuda"] is True
    assert r["full_ms"] == 2020 and r["baseline_ms"] == 20
    assert r["flips_per_ns"] == pytest.approx(32 ** 3 * 1000 / (2.0 * 1e9))       # differenced
    assert r["flips_per_ns_reported"] == pytest.approx(32 ** 3 * 1010 / (2.02 * 1e9))
    assert r["wall_full_s"] == pytest.approx(2.3)


def test_parse_log_missing_marker_and_error():
    assert B.parse_log("Started Python script\nFinished\n") is None
    bad = B.MARKER + " " + json.dumps({"runs": [], "error": "Traceback ..."})
    assert B.parse_log(bad)["error"].startswith("Traceback")
    with pytest.raises(RuntimeError):
        B.rates(B.parse_log(bad), 10)
    nocuda = json.loads(SAMPLE.split(B.MARKER + " ")[1].split("\n")[0])
    nocuda["cuda"] = "\t0"
    with pytest.raises(RuntimeError):
        B.rates(nocuda, 10)


# ------------------------------------------------------------------ build recipe
def test_build_commands():
    cmds = B.build_commands("80", 1, "3.10", "12.4", jobs=16)
    assert cmds[0] == ["make", "configure", "arch=80", "sprec=1", "python=3.10", "cuda=12.4"]
    assert cmds[1] == ["make", "compile", "-j", "16"]
    assert cmds[2] == ["make", "install"]
    assert B.build_commands("90", 0, "3.10", "12.4", jobs=4)[0][2:4] == ["arch=90", "sprec=0"]


def test_cuda_version_and_python_detection(tmp_path):
    nvcc = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.4, V12.4.131\n"
    assert B.cuda_version(nvcc) == "12.4"
    assert B.cuda_version("garbage") is None
    for v in ("3.8", "3.10"):
        (tmp_path / f"python{v}").mkdir()
        (tmp_path / f"python{v}" / "Python.h").write_text("")
    (tmp_path / "python3.12").mkdir()                                       # no header: ignored
    assert B.python_version(tmp_path) == "3.10"
    assert B.python_version(tmp_path / "nothing") is None


def test_build_dir_name_per_precision_and_arch(tmp_path):
    assert B.build_dir(tmp_path, "80", "single") == tmp_path / "boris-sp-arch80"
    assert B.build_dir(tmp_path, "90", "double") == tmp_path / "boris-dp-arch90"
    with pytest.raises(ValueError):
        B.build_dir(tmp_path, "80", "half")


def test_build_is_idempotent_when_exe_exists(tmp_path, monkeypatch):
    d = B.build_dir(tmp_path, "80", "single")
    d.mkdir()
    (d / "BorisLin").write_text("")

    def boom(*a, **k):
        raise AssertionError("no subprocess when the executable is cached")
    monkeypatch.setattr(B.subprocess, "run", boom)
    assert B.build_boris(tmp_path, "80", "single") == d / "BorisLin"


# ------------------------------------------------------------------ pairing of batches and rows
def test_precision_for_batch():
    assert B.precision_for_batch(1) == "single"
    assert B.precision_for_batch(128) == "single"
    assert B.precision_for_batch("1d") == "double"
    assert B.precisions_needed((1, "1d")) == ["single", "double"]
    assert B.precisions_needed((1, 128)) == ["single"]


def test_assemble_row_with_fake_timings():
    ours = {1: [2.0, 2.2, 2.1], "1d": [1.0, 1.1, 1.05]}
    boris = {"single": [0.5, 0.55, 0.52], "double": [0.4, 0.42, 0.41]}
    row = B.assemble_row("sc", 32, 32 ** 3, ours, boris, runs={"single": [], "double": []}, sweeps={"single": 100,
                         "double": 80}, pilots={}, batches=(1, "1d"))
    assert row["system"] == "Heisenberg sc" and row["cells"] == 32 and row["active_sites"] == 32 ** 3
    assert row["boris_flips_per_ns"] == boris
    assert row["ours_flips_per_ns"] == {"1": ours[1], "1d": ours["1d"]}
    assert row["paired_boris_precision"] == {"1": "single", "1d": "double"}
    m1 = row["log_ratio_B1"]
    assert m1["mean"] == pytest.approx(np.mean(np.log(np.array(ours[1]) / np.array(boris["single"]))))
    assert m1["low"] <= m1["mean"] <= m1["high"]
    md = row["log_ratio_B1d"]
    assert md["mean"] == pytest.approx(np.mean(np.log(np.array(ours["1d"]) / np.array(boris["double"]))))
    assert math.isfinite(md["low"]) and math.isfinite(md["high"])
