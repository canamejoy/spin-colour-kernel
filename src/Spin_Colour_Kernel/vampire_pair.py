"""E6vamp: paired throughput of the generated kernel and VAMPIRE's CUDA Monte Carlo on the Heisenberg ferromagnet.

VAMPIRE (https://github.com/richard-evans/vampire, GPL-2.0) runs Metropolis Monte Carlo on the GPU in
src/cuda/monte_carlo.cu: a greedy graph colouring of the neighbour list (colour_split), then one kernel launch
per colour per step. VAMPIRE is cloned and built at run time; no VAMPIRE source is vendored here.

Model mapping. Ours: E = -J sum_<ij> S_i . S_j (each bond once, |S| = 1), acceptance u < exp(-dE / T) with T in
units of J / kB. VAMPIRE: per-spin exchange energy -J_ij sum_j S_i . S_j with J_ij in J/link
(exchange/energy.cpp), the CUDA kernel computes dE = -mu_s (S_new - S_old) . h with h = sum_j (J_ij / mu_s) S_j
and accepts u < exp(-dE / (kB T_K)) with kB = 1.38064852e-23 J/K (monte_carlo.cu). Both dE are identical, so
the reduced temperature T maps to T_K = T * J_LINK / KB; mu_s cancels.

Move. montecarlo:algorithm = uniform: three normal deviates per atom, normalised (a new direction uniform on
the sphere, independent of the old one), the same proposal as ours (generator.py, proposal (9)).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

#: VAMPIRE main branch head on 2026-11-10 (`git ls-remote`, refs/heads/main; the repository has no master)
VAMPIRE_COMMIT = "525bc27ee44c525aee229570f30f3d4c61d54f66"
#: exchange per link, J (any positive value works: only J / (kB T_K) enters)
J_LINK = 1.0e-21
#: Boltzmann constant as written in VAMPIRE's CUDA Monte Carlo kernel (dE / (1.38064852e-23 * T))
KB = 1.38064852e-23
#: VAMPIRE's default lattice constant; irrelevant for nearest-neighbour exchange without dipolar fields
UNIT_CELL_A = 3.54
MAT_FILE = "heisenberg.mat"
BASIS = {"sc": 1, "bcc": 2, "fcc": 4}


def kelvin(T: float) -> float:
    """VAMPIRE temperature (K) of the reduced temperature T (units of J / kB)."""
    return T * J_LINK / KB


# ------------------------------------------------------------------ build
def build_vampire(work: Path, sm: str, precision: str = "DP") -> Path | None:
    """Clone VAMPIRE at VAMPIRE_COMMIT and build its `cuda` make target (executable `vampire-cuda`).

    Make-command-line overrides (the makefile itself is not edited):
      CUDA_ARCH   the shipped list starts at compute_35/sm_35, which CUDA 12 no longer accepts; one gencode for
                  `sm` (SASS) plus its PTX instead.
      CUDALIBS    the shipped value hard-codes /usr/local/cuda/lib64; $CUDA_HOME/lib64 instead.
      GCC_LDFLAGS the `cuda` link line passes GCC_LDFLAGS to nvcc, and the shipped value carries the gcc-only
                  -Wsign-compare that nvcc rejects as an unknown option; the same flags without it.
      CUDA_PR     -DCUDA_DP (the shipped default) or -DCUDA_SP. Single precision is a throughput reference only:
                  its moves compute `1.0 / __frsqrt_rn(r2)` = |r| and multiply by it, so trial spins are not
                  normalised in SP (monte_carlo.cu); double precision uses 1 / __dsqrt_rn and is correct.
    Source patch: cudaThreadSynchronize (deprecated since CUDA 4, gone in recent toolkits) -> cudaDeviceSynchronize,
    same semantics. If the build still fails, it retries once with -std=c++17 for nvcc in src/cuda/makefile
    (CCCL/Thrust of newer toolkits require C++17; the makefile asks for C++14)."""
    if precision not in ("DP", "SP"):
        raise ValueError("precision must be 'DP' or 'SP'")
    d = work / ("vampire" if precision == "DP" else "vampire-sp")
    exe = d / "vampire-cuda"
    if exe.exists():
        return exe
    log = ""
    try:
        if not d.exists():
            subprocess.run(["git", "clone", "-q", "https://github.com/richard-evans/vampire", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", VAMPIRE_COMMIT], check=True)
        for f in list((d / "src" / "cuda").glob("*.cu")) + list((d / "src" / "cuda").glob("*.hpp")):
            t = f.read_text()
            # CUDA 12 removed cudaThreadSynchronize and renamed cuSPARSE's CSRMV_ALG1 (same algorithm)
            t2 = t.replace("cudaThreadSynchronize", "cudaDeviceSynchronize").replace("CUSPARSE_CSRMV_ALG1",
                                                                                  "CUSPARSE_SPMV_CSR_ALG1")
            if t2 != t:
                f.write_text(t2)
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        env = dict(os.environ, PATH=f"{cuda}/bin" + os.pathsep + os.environ.get("PATH", ""))
        mk = ["make", "-C", str(d), f"-j{os.cpu_count() or 4}", "cuda",
              f"CUDA_ARCH=-gencode arch=compute_{sm},code=sm_{sm} -gencode arch=compute_{sm},code=compute_{sm}",
              f"CUDALIBS=-L{cuda}/lib64 -lcudart -lcusparse -lcurand",
              "GCC_LDFLAGS=-lstdc++ -I./hdr -I./src/qvoronoi",
              f"CUDA_PR=-DCUDA_{precision}"]
        r = subprocess.run(mk, capture_output=True, text=True, env=env)
        log = r.stdout + r.stderr
        if r.returncode or not exe.exists():
            mf = d / "src" / "cuda" / "makefile"
            mf.write_text(mf.read_text().replace("-std=c++14", "-std=c++17"))
            r = subprocess.run(mk, capture_output=True, text=True, env=env)
            log += "\n--- retry with -std=c++17 ---\n" + r.stdout + r.stderr
    except Exception as e:  # reported, never hidden
        print("VAMPIRE build failed:", repr(e), log[-3000:], flush=True)
        return None
    if not exe.exists():
        print("VAMPIRE build failed:", log[-3000:], flush=True)
        return None
    return exe


# ------------------------------------------------------------------ inputs
def write_inputs(d: Path, crystal: str, cells: int, T: float, sweeps: int, seed: int,
                 gpu: bool = True, device: int = 0) -> dict:
    """Write VAMPIRE's `input` and material file for the isotropic nearest-neighbour Heisenberg ferromagnet.

    Keywords (manual/InputFileCommandReference, MaterialFileCommandReference, and their parsers at
    VAMPIRE_COMMIT): create:crystal-structure (sc|bcc|fcc, conventional cubic cell), create:periodic-boundaries-x/y/z,
    dimensions:unit-cell-size and dimensions:system-size-x/y/z (VAMPIRE takes iceil(size / a) cells,
    create.cpp, so the size is (cells - 1/2) a and periodic boundaries round it back to cells * a),
    sim:program = time-series (equilibrate sim:equilibration-time-steps, then sim:total-time-steps),
    sim:integrator = monte-carlo, montecarlo:algorithm = uniform, gpu:acceleration, gpu:device,
    sim:temperature and sim:equilibration-temperature (K), sim:time-steps-increment = sweeps (statistics once),
    sim:integrator-random-seed (seeds the cuRAND generator); no output: or screen: keywords (minimal output).
    Material: exchange-matrix[1] in J/link, no anisotropy, no applied field (none is declared)."""
    if crystal not in BASIS:
        raise ValueError(f"crystal must be one of {sorted(BASIS)}")
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    T_K = kelvin(T)
    size = (cells - 0.5) * UNIT_CELL_A
    inp = [
        "# isotropic Heisenberg ferromagnet, generated by Spin_Colour_Kernel.vampire_pair",
        f"create:crystal-structure = {crystal}",
        "create:periodic-boundaries-x",
        "create:periodic-boundaries-y",
        "create:periodic-boundaries-z",
        f"dimensions:unit-cell-size = {UNIT_CELL_A} !A",
        f"dimensions:system-size-x = {size:.6f} !A",
        f"dimensions:system-size-y = {size:.6f} !A",
        f"dimensions:system-size-z = {size:.6f} !A",
        f"material:file = {MAT_FILE}",
        "sim:program = time-series",
        "sim:integrator = monte-carlo",
        "montecarlo:algorithm = uniform",
        f"gpu:acceleration = {'on' if gpu else 'off'}",
    ]
    if gpu:
        inp.append(f"gpu:device = {int(device)}")
    inp += [
        f"sim:temperature = {T_K!r}",
        f"sim:equilibration-temperature = {T_K!r}",
        "sim:equilibration-time-steps = 0",
        f"sim:total-time-steps = {int(sweeps)}",
        f"sim:time-steps-increment = {int(sweeps)}",
        f"sim:integrator-random-seed = {int(seed)}",
    ]
    mat = [
        "material:num-materials = 1",
        "material[1]:material-name = heisenberg",
        f"material[1]:exchange-matrix[1] = {J_LINK!r}",
        "material[1]:atomic-spin-moment = 1.72 !muB",
        "material[1]:second-order-uniaxial-anisotropy-constant = 0.0",
        "material[1]:damping-constant = 1.0",
    ]
    (d / "input").write_text("\n".join(inp) + "\n")
    (d / MAT_FILE).write_text("\n".join(mat) + "\n")
    return {"temperature_K": T_K, "n_atoms_expected": cells ** 3 * BASIS[crystal], "J_link": J_LINK, "kB": KB}


# ------------------------------------------------------------------ run and parse
def flips_per_ns(n_atoms: int, sweeps: int, seconds: float) -> float:
    """Spin updates per ns; NaN when the time is not positive."""
    if not seconds > 0:
        return float("nan")
    return n_atoms * sweeps / (seconds * 1e9)


def parse_log(text: str) -> dict:
    """Facts from VAMPIRE's stdout (print statements at VAMPIRE_COMMIT: create.cpp, cuda/initialize.cu,
    cuda/monte_carlo.cu colour_split, simulate/sim.cpp)."""
    def num(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    prec = re.search(r"CUDA has been enabled in (\w+) precision mode", text)
    colours = [int(n) for n in re.findall(r"Colour: \d+, natoms = (\d+)", text)]
    return {"n_atoms": num(r"Number of atoms generated:\s*(\d+)", int),
            "gpu": prec is not None,
            "precision": prec.group(1) if prec else None,
            "colours": colours, "n_colours": len(colours),
            "sim_seconds": num(r"Simulation run time \[s\]:\s*([0-9.eE+-]+)"),
            "total_moves": num(r"Total moves:\s*(\d+)", int),
            "accepted_pct": num(r"([0-9.eE+-]+)% Accepted")}


def _run_once(exe: Path, d: Path, timeout: float) -> tuple[dict, float, str]:
    t0 = time.perf_counter()
    r = subprocess.run([str(Path(exe).resolve())], cwd=d, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    out = r.stdout + r.stderr
    f = parse_log(out)
    if r.returncode or f["sim_seconds"] is None:
        raise RuntimeError(f"VAMPIRE exited {r.returncode} without a run time: " + out[-2000:])
    return f, wall, out


def run_vampire(exe: Path, d: Path, crystal: str, cells: int, T: float, sweeps: int, seed: int,
                baseline_sweeps: int = 10, gpu: bool = True, device: int = 0, timeout: float = 3600) -> dict:
    """Run `sweeps` and `baseline_sweeps` steps and difference the reported simulation times.

    VAMPIRE's "Simulation run time [s]" (sim.cpp stopwatch) excludes system creation but includes the lazy GPU
    Monte Carlo set-up done inside the first step (cuRAND, graph colouring, one trial step), so
    flips/ns = n_atoms (sweeps - baseline) / (t_sweeps - t_baseline). The baseline needs >= 1 step: with 0 steps
    the set-up never runs. flips_per_ns_reported = n_atoms sweeps / t_sweeps keeps the set-up in."""
    baseline_sweeps = max(1, int(baseline_sweeps))
    if sweeps <= baseline_sweeps:
        raise ValueError("sweeps must exceed baseline_sweeps")
    d = Path(d)
    res = {}
    for name, S in (("full", sweeps), ("baseline", baseline_sweeps)):
        info = write_inputs(d / name, crystal, cells, T, S, seed, gpu=gpu, device=device)
        f, wall, out = _run_once(exe, d / name, timeout)
        (d / name / "stdout.txt").write_text(out)
        res[name] = {**f, "sweeps": S, "process_wall_s": wall}
    n = res["full"]["n_atoms"] or info["n_atoms_expected"]
    dt = res["full"]["sim_seconds"] - res["baseline"]["sim_seconds"]
    return {"crystal": crystal, "cells": cells, "n_atoms": n, "n_atoms_expected": info["n_atoms_expected"],
            "temperature_K": info["temperature_K"], "sweeps": sweeps, "baseline_sweeps": baseline_sweeps,
            "sim_seconds": res["full"]["sim_seconds"], "baseline_sim_seconds": res["baseline"]["sim_seconds"],
            "flips_per_ns": flips_per_ns(n, sweeps - baseline_sweeps, dt),
            "flips_per_ns_reported": flips_per_ns(n, sweeps, res["full"]["sim_seconds"]),
            "gpu": res["full"]["gpu"], "precision": res["full"]["precision"],
            "n_colours": res["full"]["n_colours"], "colours": res["full"]["colours"],
            "accepted_pct": res["full"]["accepted_pct"], "total_moves": res["full"]["total_moves"],
            "process_wall_s": res["full"]["process_wall_s"]}


# ------------------------------------------------------------------ E6vamp
def vampire_pair(out: Path, device="cuda", systems=("sc", "bcc", "fcc"), reps: int = 5, seed: int = 6006,
                 cells: int = 32, T: float = 1.2, batches=(1, 128), target_seconds: float = 2.0,
                 work: Path = Path("/tmp/sck"), tag: str = "E6vamp", cells_list=None, warmup: int = 0):
    """Generated kernel (batch 1 and 128) against VAMPIRE's CUDA Monte Carlo (double precision, uniform move),
    classical Heisenberg ferromagnet on sc / bcc / fcc, `cells`^3 conventional cells (32768 / 65536 / 131072
    spins at 32), reduced temperature T (1.2: ordered on all three, below the Curie points of about 1.443 (sc), 2.054 (bcc) and 3.18 (fcc)), the sides alternated on the same
    device for `reps` rounds (order reversed every round); paired log speed ratios with bootstrap intervals."""
    import torch

    from . import system as sy
    from .exp_core import save
    from .exp_perf import _events_ms, _sm, boot_ci, gpu_state
    from .sampler import Sampler

    dev_index = torch.device(device).index or 0
    exe = build_vampire(work, _sm())
    res = {"vampire_commit": VAMPIRE_COMMIT, "built": exe is not None, "T_reduced": T, "J_link": J_LINK, "kB": KB,
           "temperature_K": kelvin(T), "move": "uniform (normalised Gaussian triple), Metropolis",
           "vampire_precision_build": "CUDA_DP", "batches": list(batches), "reps": reps, "seed": seed,
           "cells": cells, "rows": []}
    if exe is None:
        res["status"] = "VAMPIRE not built"
        return save(out, tag, res)
    rundir = work / "vampire-runs"
    res["cells_list"] = list(cells_list or (cells,))
    res["warmup_rounds_discarded"] = warmup
    for cells, crystal in [(c, x) for c in (cells_list or (cells,)) for x in systems]:
        s = sy.heisenberg(crystal, cells)
        n = s.n_active
        prep = {}
        for B in batches:
            # "1d": batch 1 in double precision, to compare with VAMPIRE at equal precision (v11)
            Bn, dt = int(str(B).rstrip("d")), (torch.float64 if str(B).endswith("d") else None)
            A = Sampler(s, Bn, device, seed, spin_dtype=dt)
            A.set_temperature(T)
            S = A.init_spins(seed=seed)
            k = max(1, int(math.ceil(2e7 / (Bn * n))))
            replay = A.graph(S, k)
            for _ in range(2):
                replay()
            ms0 = _events_ms(replay, 1, A.device)
            nrep = max(1, int(math.ceil(100.0 / ms0)))
            prep[B] = (replay, Bn * n * k * nrep, A.device, nrep, A, S)
        # pilot: size VAMPIRE's run to ~target_seconds of stepping beyond the set-up
        pil = run_vampire(exe, rundir / f"{crystal}{cells}" / "pilot", crystal, cells, T, 110, seed, baseline_sweeps=10,
                          device=dev_index)
        per_step = (pil["sim_seconds"] - pil["baseline_sim_seconds"]) / 100.0
        sweeps = int(min(50_000, max(200, round(target_seconds / per_step)))) if per_step > 0 else 1000
        ours = {B: [] for B in batches}
        vamp, vrows = [], []
        for i in range(-warmup, reps):           # negative rounds are warm-up, discarded
            def run_ours():
                for B in batches:
                    replay, upd, dv, nrep = prep[B][:4]
                    v = upd / (_events_ms(replay, nrep, dv) * 1e6)
                    if i >= 0:
                        ours[B].append(v)

            def run_vamp():
                torch.cuda.synchronize(device)
                v = run_vampire(exe, rundir / f"{crystal}{cells}" / f"rep{i}", crystal, cells, T, sweeps, seed + i,
                                device=dev_index)
                if i >= 0:
                    vamp.append(v["flips_per_ns"])
                    vrows.append(v)
            for fn in ((run_ours, run_vamp) if i % 2 == 0 else (run_vamp, run_ours)):
                fn()
        row = {"system": f"Heisenberg {crystal}", "cells": cells, "active_sites": n, "vampire_atoms": vrows[0]["n_atoms"],
               "atoms_match": vrows[0]["n_atoms"] == n, "vampire_sweeps": sweeps,
               "vampire_flips_per_ns": vamp, "vampire_runs": vrows, "vampire_pilot": pil,
               "ours_flips_per_ns": {str(B): ours[B] for B in batches}, "gpu_state": gpu_state(device),
               "order": "alternated, reversed every round"}
        for B in batches:
            m, lo, hi = boot_ci(np.log(np.asarray(ours[B]) / np.asarray(vamp)))
            row[f"log_ratio_B{B}"] = {"mean": m, "low": lo, "high": hi}
            print(f"E6vamp {crystal:3s} {cells}^3 n={n} ours B={B} {np.mean(ours[B]):.3f}/ns  VAMPIRE {np.mean(vamp):.4f}/ns "
                  f"({vrows[0]['n_colours']} colours)  log ratio {m:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
        res["rows"].append(row)
        del prep
        torch.cuda.empty_cache()
    res["status"] = "run"
    return save(out, tag, res)
