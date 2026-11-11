"""E6boris: paired throughput of the generated kernel and BORIS's GPU atomistic Monte Carlo on the Heisenberg ferromagnet.

BORIS (https://github.com/SerbanL/BORIS, multi-GPU upgrade of Boris v3.8, GPL-3.0) is cloned at BORIS_COMMIT and
built at run time with its own makefile; no BORIS source is vendored here. Line references below are to that commit.

Crystals. Only the simple cubic atomistic mesh exists on the GPU: Boris/CompileFlags.h defines
MESH_COMPILATION_ATOM_CUBIC and leaves MESH_COMPILATION_ATOM_BCC commented out, Atom_Mesh_BCC has no CUDA
counterpart (there is no Atom_Mesh_BCCCUDA*), and there is no fcc atomistic mesh at all. E6boris therefore pairs
on sc only; bcc and fcc are recorded as unsupported, not skipped silently.

Model mapping. Ours: E = -J sum_<ij> S_i . S_j (each bond once, |S| = 1), acceptance u < exp(-dE / T) with T in
units of J / kB. BORIS: Atom_ExchangeCUDA.cu Get_Atomistic_EnergyChange_SC_ExchangeCUDA returns
dE = -J (S_new/|S_new| - S_old/|S_old|) . sum_nn S_j/|S_j| with J in J (setparam J), and
Atom_Mesh_CubicCUDA_MonteCarlo.cu accepts when u <= exp(-dE / (BOLTZMANN T_K)) with BOLTZMANN = 1.3806488e-23
(BorisLib/Funcs_Math_base.h). Both dE are identical, so T maps to T_K = T * J_LINK / KB; mu_s cancels.
Demagnetising (atomistic `dipoledipole`) and `Zeeman`, the two default modules besides `exchange`
(Atom_Mesh_Cubic.cpp), are deleted, so exchange is the only energy term.

Move (differs from ours). BORIS's classical parallel Monte Carlo (mcserial 0, mcconstrain 0; the only CUDA path) is
a red/black checkerboard over the sc lattice, one kernel per colour per iteration, every spin proposed once per
iteration. The trial spin is the old one rotated by a polar angle theta = u1 * cone and an azimuth phi = 2 pi u2
(relrotate_polar): a cone move, symmetric (it depends only on the angle between old and new) but not uniform on
the sphere, and the cone angle adapts by +-1 degree per iteration towards a 0.5 acceptance rate (MeshBaseMonteCarlo.cpp
MonteCarlo_AdaptiveAngle, limits 1..180 degrees, start 30, Boris_Enums_Defs.h MONTECARLO_*). Ours proposes a new
direction uniform on the sphere, independent of the old one. The rates below are spin updates per ns for both;
they are not equal-acceptance or equal-decorrelation rates.

Timing. BORIS's own `benchtime` command (Commands.cpp CMD_BENCHTIME) returns sim_end_ms - sim_start_ms, which
starts after module initialisation (SimControl.cpp PrepareRunSimulation) and ends at StopSimulation, at millisecond
resolution. Each BORIS process runs a warm-up stage, a baseline stage and a full stage (reset between them), and
flips/ns = n (full - baseline) / (t_full - t_baseline), as for VAMPIRE in vampire_pair.

Precision. The makefile builds single (sprec=1) or double (sprec=0) CUDA arithmetic. Batch 1 and 128 (ours in
float32) pair with the single-precision build, "1d" (ours in float64) with the double-precision build.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from .vampire_pair import J_LINK, flips_per_ns

#: BORIS multi-GPU head on 2026-11-10 (2023-11-02, `git ls-remote`, default branch)
BORIS_COMMIT = "048503d2ca6475e91b073eb1762ecf9d3634c3e1"
BORIS_REPO = "https://github.com/SerbanL/BORIS"
#: Boltzmann constant as BORIS defines it (BorisLib/Funcs_Math_base.h BOLTZMANN)
KB = 1.3806488e-23
#: lattice constant (m); irrelevant without dipolar fields, only n = round(size / a) matters
LATTICE_A = 2.5e-10
MESH = "asc"
SUPPORTED_CRYSTALS = ("sc",)
UNSUPPORTED = {"bcc": "atomistic bcc mesh not compiled (CompileFlags.h) and without a CUDA implementation",
               "fcc": "no fcc atomistic mesh in BORIS"}
RESULT_FILE = "boris_result.json"
SCRIPT_FILE = "pair.py"
MARKER = "BORIS_PAIR_RESULT"
EXE = "BorisLin"
PRECISIONS = {"single": 1, "double": 0}

MODEL_NOTE = (
    "Heisenberg sc, nearest-neighbour exchange only (BORIS modules dipoledipole and Zeeman deleted), periodic in x, y, z. "
    "Same dE = -J (S_new - S_old) . sum_nn S_j on both sides; T_K = T J / kB with BORIS's kB = 1.3806488e-23. "
    "Moves differ: BORIS proposes a cone rotation (theta = u cone, phi = 2 pi u) with the cone adapted towards 0.5 "
    "acceptance, red/black checkerboard; ours proposes a direction uniform on the sphere. Rates are spin updates per ns. "
    "bcc and fcc are not available on BORIS's GPU path (only Atom_Mesh_Cubic has CUDA code).")


def kelvin(T: float) -> float:
    """BORIS temperature (K) of the reduced temperature T (units of J / kB)."""
    return T * J_LINK / KB


def arch_for_capability(cap) -> str:
    """BORIS's `arch`: Simulation.cpp selects a GPU only when major * 100 == __CUDA_ARCH__, and `make configure`
    writes __CUDA_ARCH__ = arch * 10, so arch = major * 10 (sm_80 code also runs on 8.6 / 8.9)."""
    return f"{int(cap[0])}0"


def precision_for_batch(B) -> str:
    return "double" if str(B).endswith("d") else "single"


def precisions_needed(batches) -> list:
    out = []
    for B in batches:
        p = precision_for_batch(B)
        if p not in out:
            out.append(p)
    return out


def n_spins(crystal: str, cells: int) -> int:
    _check(crystal)
    return cells ** 3


def _check(crystal: str):
    if crystal not in SUPPORTED_CRYSTALS:
        raise ValueError(f"BORIS GPU Monte Carlo supports {SUPPORTED_CRYSTALS}, not {crystal!r}: "
                         + UNSUPPORTED.get(crystal, "unknown crystal"))


# ------------------------------------------------------------------ build
def build_dir(root: Path, arch: str, precision: str) -> Path:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    return Path(root) / f"boris-{precision[0]}p-arch{arch}"


def build_commands(arch: str, sprec: int, python: str, cuda: str, jobs: int) -> list:
    """The makefile's three targets (makefile at BORIS_COMMIT: configure, compile, install)."""
    return [["make", "configure", f"arch={arch}", f"sprec={int(sprec)}", f"python={python}", f"cuda={cuda}"],
            ["make", "compile", "-j", str(int(jobs))],
            ["make", "install"]]


def cuda_version(nvcc_out: str):
    m = re.search(r"release (\d+\.\d+)", nvcc_out or "")
    return m.group(1) if m else None


def python_version(include_root: Path = Path("/usr/include")):
    """Highest python3.X under include_root with a Python.h (the makefile compiles with -I/usr/include/pythonX.Y
    and links -lpythonX.Y, so the system Python's dev package is needed, not Modal's add_python build)."""
    vs = []
    for p in Path(include_root).glob("python3.*"):
        m = re.fullmatch(r"python(3\.\d+)", p.name)
        if m and (p / "Python.h").exists():
            vs.append(m.group(1))
    return max(vs, key=lambda v: int(v.split(".")[1])) if vs else None


#: Linux portability patch: BORIS defines atomicAdd(unsigned long *, unsigned long) (size_t under g++) only for
#: __CUDA_ARCH__ <= 800 (BorisCUDALib/atomics.cuh:163); on sm_90 (H100) BorisCUDALib/Reduction.cuh's
#: atomicAdd(size_t *, size_t) has no overload. The patch fills that gap only above sm_80, with BORIS's own body.
SIZE_T_ATOMIC = """
// spin-colour-kernel: Linux portability (BORIS defines this overload only for __CUDA_ARCH__ <= 800)
#if __CUDA_ARCH__ > 800
inline __device__ void atomicAdd(unsigned long* address, unsigned long val)
{
\tatomicAdd((unsigned long long int*)address, (unsigned long long)val);
}
#endif
"""


def patch_sources(d: Path) -> list:
    """Apply the Linux portability patches once; return what was patched (recorded in build_info.json)."""
    done = []
    f = d / "BorisCUDALib" / "atomics.cuh"
    t = f.read_text()
    if "spin-colour-kernel: Linux portability" not in t:
        f.write_text(t + SIZE_T_ATOMIC)
        done.append("BorisCUDALib/atomics.cuh: atomicAdd(unsigned long *, unsigned long) for __CUDA_ARCH__ > 800")
    return done


def build_boris(root: Path, arch: str, precision: str = "single") -> Path | None:
    """Clone BORIS at BORIS_COMMIT into root/boris-{s,d}p-arch{arch} and build BorisLin; idempotent (returns the cached
    executable). Uses the makefile unchanged; only if the build fails it retries once with -std=c++17 for nvcc (newer
    CCCL/Thrust warn on or reject C++14). The repository's CompileFlags.h already sets OS_LIN, GRAPHICS 0,
    COMPILECUDA 1, PYTHON_EMBEDDING 1, MONTE_CARLO 1, ATOMISTIC 1. Writes build_info.json and build.log beside it."""
    d = build_dir(root, arch, precision)
    exe = d / EXE
    if exe.exists():
        return exe
    log = ""
    info = {"commit": BORIS_COMMIT, "arch": arch, "precision": precision, "sprec": PRECISIONS[precision]}
    try:
        if not d.exists():
            subprocess.run(["git", "clone", "-q", BORIS_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", BORIS_COMMIT], check=True)
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        env = dict(os.environ, PATH=f"{cuda}/bin" + os.pathsep + os.environ.get("PATH", ""))
        nv = subprocess.run([f"{cuda}/bin/nvcc", "--version"], capture_output=True, text=True, env=env).stdout
        info["cuda"], info["python"] = cuda_version(nv), python_version()
        if info["cuda"] is None or info["python"] is None:
            raise RuntimeError(f"nvcc or Python headers missing: {info}")
        info["make"] = build_commands(arch, PRECISIONS[precision], info["python"], info["cuda"], os.cpu_count() or 4)
        info["source_patches"] = patch_sources(d)
        for attempt in range(2):
            ok = True
            for c in info["make"]:
                r = subprocess.run(c, cwd=d, capture_output=True, text=True, env=env)
                log += f"\n$ {' '.join(c)}\n" + r.stdout[-20000:] + r.stderr[-20000:]
                if r.returncode:
                    ok = False
                    break
            if ok and exe.exists():
                break
            if attempt == 0:
                mf = d / "makefile"
                mf.write_text(mf.read_text().replace("-rdc=true -c -std=c++14", "-rdc=true -c -std=c++17"))
                info["nvcc_std_patch"] = "c++14 -> c++17"
                log += "\n--- retry with nvcc -std=c++17 ---\n"
    except Exception as e:  # reported, never hidden
        log += "\n" + repr(e)
    (d / "build.log").parent.mkdir(parents=True, exist_ok=True)
    (d / "build.log").write_text(log)
    if not exe.exists():
        print("BORIS build failed:", log[-3000:], flush=True)
        return None
    (d / "Boris_Data" / "Simulations").mkdir(parents=True, exist_ok=True)   # GetUserDocumentsPath = exe dir on Linux
    (d / "build_info.json").write_text(json.dumps(info, indent=2))
    return exe


# ------------------------------------------------------------------ console commands and script
def setup_commands(crystal: str, cells: int, T: float, seed: int) -> list:
    """BORIS console commands (usages in Simulation.cpp at BORIS_COMMIT) for the isotropic nearest-neighbour Heisenberg
    ferromagnet on cells^3 sc sites, classical parallel Monte Carlo on the GPU."""
    _check(crystal)
    L = cells * LATTICE_A
    return [
        f"setameshcubic {MESH} 0 0 0 {L!r} {L!r} {L!r}",   # single sc atomistic mesh, rectangle in m
        f"cellsize {MESH} {LATTICE_A!r}",                  # n = round(size / a) (Atom_Mesh_Cubic.cpp)
        f"pbc {MESH} x 1", f"pbc {MESH} y 1", f"pbc {MESH} z 1",
        f"delmodule {MESH} dipoledipole",                  # default modules: dipoledipole, exchange, Zeeman
        f"delmodule {MESH} Zeeman",
        f"setparam {MESH} J {J_LINK!r}",
        f"temperature {MESH} {kelvin(T)!r}",
        f"prngseed {MESH} {int(seed)}",
        "setstage MonteCarlo",
        f"mcserial {MESH} 0",                              # parallel red/black (serial is CPU only)
        f"mcconstrain {MESH} 0",                           # classical, not constrained Monte Carlo
        "mccomputefields 0",                               # no field evaluation after each MC iteration
        "iterupdate 0",                                    # no display / NVML temperature poll in the loop
        "cuda 1",
    ]


_SCRIPT = """# generated by Spin_Colour_Kernel.boris_pair for BORIS {commit}; runs inside BorisLin -s (embedded Python)
import json, time, traceback
import BPython
P = {params!r}
out = {{"params": P["params"], "setup": [], "runs": []}}
try:
    for c in P["setup"]:
        out["setup"].append([c, BPython.BPython_Message("*" + c)])
    out["cuda"] = BPython.BPython_Message("*cuda")
    for name, iters in P["schedule"]:
        BPython.BPython_Message("*reset")
        BPython.BPython_Message("*editstagestop 0 iter %d" % iters)
        t0 = time.perf_counter()
        BPython.BPython_Message_Run()
        wall = time.perf_counter() - t0
        out["runs"].append({{"name": name, "iters": iters, "wall_s": wall, "benchtime": BPython.BPython_Message("*benchtime")}})
    out["mcparams"] = BPython.BPython_Message("*showdata MCparams")
except Exception:
    out["error"] = traceback.format_exc()
fh = open(P["result"], "w")
fh.write(json.dumps(out))
fh.close()
print(P["marker"] + " " + json.dumps(out))
"""


def write_script(d, crystal: str, cells: int, T: float, seed: int, warm: int, baseline: int, full: int) -> Path:
    """Write the embedded-Python script BORIS runs with `BorisLin -g <dev> -s <script>`. It talks to BORIS through
    the BPython module directly (Boris/PythonScripting.h: BPython_Message with a '*' prefix returns tab-led fields,
    BPython_Message_Run blocks until the stage schedule ends), so NetSocks.py and its numpy/matplotlib imports are
    not needed. BORIS kills itself with SIGTERM when the script ends (Commands_PythonScripting.cpp)."""
    _check(crystal)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    params = {"params": {"crystal": crystal, "cells": cells, "T": T, "T_K": kelvin(T), "seed": seed},
              "setup": setup_commands(crystal, cells, T, seed),
              "schedule": [["warm", int(warm)], ["baseline", int(baseline)], ["full", int(full)]],
              "result": str((d / RESULT_FILE).resolve()), "marker": MARKER}
    p = d / SCRIPT_FILE
    p.write_text(_SCRIPT.format(commit=BORIS_COMMIT, params=params))
    return p


# ------------------------------------------------------------------ parse and rates
def ret_number(s):
    """First numeric field of a BPython_Message return ('\\t2020' -> 2020.0); None when empty."""
    if isinstance(s, (int, float)):
        return float(s)
    for f in (s or "").split("\t"):
        try:
            return float(f)
        except ValueError:
            continue
    return None


def parse_log(text: str):
    """The script's result dict from BORIS's console output (the MARKER line), or None."""
    for line in (text or "").splitlines():
        if line.startswith(MARKER + " "):
            return json.loads(line[len(MARKER) + 1:])
    return None


def rates(res, n_atoms: int) -> dict:
    if res is None:
        raise RuntimeError("BORIS produced no result")
    if res.get("error"):
        raise RuntimeError("BORIS script failed: " + res["error"][-2000:])
    if ret_number(res.get("cuda")) != 1:
        raise RuntimeError(f"BORIS did not enable CUDA (cuda -> {res.get('cuda')!r})")
    runs = {r["name"]: r for r in res["runs"]}
    f, b = runs["full"], runs["baseline"]
    tf, tb = ret_number(f["benchtime"]), ret_number(b["benchtime"])
    return {"cuda": True, "full_ms": tf, "baseline_ms": tb, "full_iters": f["iters"], "baseline_iters": b["iters"],
            "flips_per_ns": flips_per_ns(n_atoms, f["iters"] - b["iters"], (tf - tb) / 1e3),
            "flips_per_ns_reported": flips_per_ns(n_atoms, f["iters"], tf / 1e3),
            "wall_full_s": f["wall_s"], "wall_baseline_s": b["wall_s"],
            "warm_ms": ret_number(runs["warm"]["benchtime"]) if "warm" in runs else None,
            "mcparams": res.get("mcparams")}


def run_boris(exe: Path, d: Path, crystal: str, cells: int, T: float, full: int, seed: int, baseline: int = 10,
              warm: int = 20, device: int = 0, timeout: float = 3600) -> dict:
    """One BorisLin process: warm-up, baseline and full Monte Carlo stages; rates from BORIS's benchtime."""
    if full <= baseline:
        raise ValueError("full must exceed baseline")
    d = Path(d)
    script = write_script(d, crystal, cells, T, seed, warm, baseline, full)
    res_file = d / RESULT_FILE
    if res_file.exists():
        res_file.unlink()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}   # embedded system Python
    exe = Path(exe).resolve()
    t0 = time.perf_counter()
    with open(d / "stdout.txt", "w") as log:
        # stdin stays open and silent: at EOF the console thread would spin on getline (BorisConsole.cpp TextInput)
        p = subprocess.Popen([str(exe), "-g", str(int(device)), "-s", str(script.resolve())], cwd=exe.parent,
                             stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        finally:
            p.stdin.close()
    wall = time.perf_counter() - t0
    out = (d / "stdout.txt").read_text(errors="replace")
    res = json.loads(res_file.read_text()) if res_file.exists() else parse_log(out)
    if res is None:
        raise RuntimeError(f"BorisLin exited {rc} without a result: " + out[-2000:])
    r = rates(res, n_spins(crystal, cells))
    return {"crystal": crystal, "cells": cells, "n_atoms": n_spins(crystal, cells), "temperature_K": kelvin(T),
            "returncode": rc, "process_wall_s": wall, **r}


def assemble_row(crystal, cells, n, ours: dict, boris: dict, runs: dict, sweeps: dict, pilots: dict, batches,
                 gpu=None) -> dict:
    from .exp_perf import boot_ci

    row = {"system": f"Heisenberg {crystal}", "cells": cells, "active_sites": n, "boris_atoms": n,
           "boris_sweeps": sweeps, "boris_flips_per_ns": boris, "boris_runs": runs, "boris_pilot": pilots,
           "ours_flips_per_ns": {str(B): list(ours[B]) for B in batches},
           "paired_boris_precision": {str(B): precision_for_batch(B) for B in batches},
           "gpu_state": gpu or {}, "order": "alternated, reversed every round"}
    for B in batches:
        m, lo, hi = boot_ci(np.log(np.asarray(ours[B]) / np.asarray(boris[precision_for_batch(B)])))
        row[f"log_ratio_B{B}"] = {"mean": m, "low": lo, "high": hi}
    return row


# ------------------------------------------------------------------ our side (same measurement as vampire_pair)
def _prep_ours(s, B, device, seed, T):
    import torch

    from .exp_perf import _events_ms
    from .sampler import Sampler

    Bn, dt = int(str(B).rstrip("d")), (torch.float64 if str(B).endswith("d") else None)
    A = Sampler(s, Bn, device, seed, spin_dtype=dt)
    A.set_temperature(T)
    S = A.init_spins(seed=seed)
    k = max(1, int(math.ceil(2e7 / (Bn * s.n_active))))
    replay = A.graph(S, k)
    for _ in range(2):
        replay()
    ms0 = _events_ms(replay, 1, A.device)
    nrep = max(1, int(math.ceil(100.0 / ms0)))
    return (replay, Bn * s.n_active * k * nrep, A.device, nrep, A, S)


def _ours_rate(prep) -> float:
    from .exp_perf import _events_ms

    replay, upd, dv, nrep = prep[:4]
    return upd / (_events_ms(replay, nrep, dv) * 1e6)


# ------------------------------------------------------------------ E6boris
def boris_pair(out: Path, device="cuda:0", reps: int = 10, batches=(1, "1d"), cells_list=(32, 48, 64), warmup: int = 1,
               tag: str = "E6boris", T: float = 1.2, seed: int = 6106, target_seconds: float = 2.0,
               work: Path = Path("/tmp/sck"), crystals=("sc", "bcc", "fcc")):
    """Generated kernel (batch 1 float32 and "1d" float64, optionally 128) against BORIS's CUDA atomistic Monte Carlo
    (single-precision build for float32 batches, double-precision build for "1d"), Heisenberg sc, cells^3 sites,
    reduced temperature T, the sides alternated on the same device for `reps` measured rounds after `warmup`
    discarded rounds (order reversed every round); paired log speed ratios with bootstrap intervals.
    Crystals BORIS cannot run on the GPU (bcc, fcc) are listed under `unsupported_crystals`."""
    import torch

    from . import system as sy
    from .exp_core import save
    from .exp_perf import gpu_state

    dev_index = torch.device(device).index or 0
    arch = arch_for_capability(torch.cuda.get_device_capability(dev_index))
    precs = precisions_needed(batches)
    exes = {p: build_boris(work, arch, p) for p in precs}
    info = {}
    for p, e in exes.items():
        f = (e.parent / "build_info.json") if e else None
        info[p] = json.loads(f.read_text()) if f and f.exists() else {"built": False}
    run_crystals = [c for c in crystals if c in SUPPORTED_CRYSTALS]
    res = {"boris_commit": BORIS_COMMIT, "boris_repo": BORIS_REPO, "built": all(e is not None for e in exes.values()),
           "build": info, "arch": arch, "T_reduced": T, "J_link": J_LINK, "kB": KB, "temperature_K": kelvin(T),
           "move": "BORIS: cone rotation, adaptive cone angle (target acceptance 0.5), red/black checkerboard, Metropolis; "
                   "ours: uniform on the sphere, Metropolis",
           "model_note": MODEL_NOTE, "timing": "BORIS benchtime (ms), full minus baseline stage; ours CUDA events",
           "batches": [str(B) for B in batches], "paired_boris_precision": {str(B): precision_for_batch(B) for B in batches},
           "reps": reps, "seed": seed, "cells_list": list(cells_list), "warmup_rounds_discarded": warmup,
           "crystals": run_crystals,
           "unsupported_crystals": {c: UNSUPPORTED.get(c, "unknown") for c in crystals if c not in SUPPORTED_CRYSTALS},
           "rows": []}
    if not res["built"]:
        res["status"] = "BORIS not built"
        return save(out, tag, res)
    rundir = work / "boris-runs"
    for cells, crystal in [(c, x) for c in cells_list for x in run_crystals]:
        s = sy.heisenberg(crystal, cells)
        n = s.n_active
        if n != n_spins(crystal, cells):
            raise RuntimeError(f"active sites {n} != BORIS sites {n_spins(crystal, cells)}")
        prep = {B: _prep_ours(s, B, device, seed, T) for B in batches}
        pilots, sweeps = {}, {}
        for p in precs:   # size BORIS's full stage to ~target_seconds of stepping
            pil = run_boris(exes[p], rundir / f"{crystal}{cells}" / p / "pilot", crystal, cells, T, 110, seed,
                            baseline=10, warm=10, device=dev_index)
            per = (pil["full_ms"] - pil["baseline_ms"]) / 1e3 / 100.0
            pilots[p] = pil
            sweeps[p] = int(min(200_000, max(200, round(target_seconds / per)))) if per > 0 else 2000
        ours = {B: [] for B in batches}
        bor = {p: [] for p in precs}
        bruns = {p: [] for p in precs}
        for i in range(-warmup, reps):   # negative rounds are warm-up, discarded
            def run_ours():
                for B in batches:
                    v = _ours_rate(prep[B])
                    if i >= 0:
                        ours[B].append(v)

            def run_boris_side():
                torch.cuda.synchronize(device)
                for p in precs:
                    v = run_boris(exes[p], rundir / f"{crystal}{cells}" / p / f"rep{i}", crystal, cells, T, sweeps[p],
                                  seed + i, device=dev_index)
                    if i >= 0:
                        bor[p].append(v["flips_per_ns"])
                        bruns[p].append(v)
            for fn in ((run_ours, run_boris_side) if i % 2 == 0 else (run_boris_side, run_ours)):
                fn()
        row = assemble_row(crystal, cells, n, ours, bor, bruns, sweeps, pilots, batches, gpu_state(device))
        for B in batches:
            lr = row[f"log_ratio_B{B}"]
            p = precision_for_batch(B)
            print(f"E6boris {crystal} {cells}^3 n={n} ours B={B} {np.mean(ours[B]):.3f}/ns  BORIS-{p} "
                  f"{np.mean(bor[p]):.4f}/ns  log ratio {lr['mean']:+.3f} [{lr['low']:+.3f},{lr['high']:+.3f}]", flush=True)
        res["rows"].append(row)
        del prep
        torch.cuda.empty_cache()
    res["status"] = "run"
    return save(out, tag, res)
