"""E6metro: paired throughput of the generated kernel and CUDA-METRO's pyCUDA Metropolis on the 2D Heisenberg ferromagnet.

CUDA-METRO (https://github.com/arkavo/CUDA-METRO, Apache-2.0) runs Metropolis Monte Carlo for 2D magnets in pyCUDA.
Each round draws P sites ("Blocks") with replacement, evaluates every proposal against the same lattice in one launch
(one site per block, 2 threads: old and new local energy) and writes them back in a second launch (cp_grid), so the
P updates are synchronous. Its benchmarks/README.md fits the deviation from single-update Metropolis as
eps ~ 0.0276 P/N (ordered phase). CUDA-METRO is cloned at run time; no CUDA-METRO source is vendored here.

Model mapping (src/cudametro/montecarlo.py and construct.py at CUDAMETRO_COMMIT). The material row holds the spin
magnitude, J1..J4 (meV), anisotropic exchange K1..K4, single-ion A, the neighbour-shell signature and Tc; only
spin = 1 and J1 = J_MEV are non-zero here, B = 0, DMI off. The dm0 Hamiltonians return the local energy
-(J1 sum_j S_i . S_j + ...) and the kernel accepts u < exp(-dE * beta) with beta = 1 / (T_K * 8.6173e-2) in 1/meV,
so dE * beta = dE / T in our reduced units when T_K = T * J_MEV / KB_MEV. Proposal: a new direction uniform on the
sphere (uvec_processor: phi = 2 pi u, theta = acos(2 v - 1)), independent of the old one, as ours (generator.py,
proposal (9)). Their random numbers are drawn by pycuda's XORWOW generator before the loop and are not timed;
ours draw in-kernel and are timed, so the comparison is conservative for us.

Lattices. Square = the 4_4_4_8 kernels, triangular = 6_6_6_12, both one site per cell of an L x L grid indexed
row * L + col with the column wrap into the next row (helical boundaries; same N = L^2 as ours). The 3_6_3_6 family
("honeycomb") stores one site per cell with first neighbours (+1, 0), (-1, +1), (0, -1): the relation is not
symmetric, so it is no honeycomb graph and is not paired. Observed at the pinned commit: the square dm0 kernel takes
an NVEC argument that construct.run_mc_tc_4448 does not pass (passed here, unused by the dm0 Hamiltonian), and
hamiltonian_tc_2d_4_4_4_8_dm0 loops 6 / 6 / 6 / 12 times over neighbour arrays filled with 4 / 4 / 4 / 8 entries, so
two first-shell slots read uninitialised indices weighted by J1. The worker reports grid sanity (finite, unit norm)
after every timed run; triangular has no such defect.

Timing (as benchmarks/bench_speed.py): pycuda.autoinit context in a worker subprocess (isolates CUDA-METRO's context
and any fault from ours), CUDA events around `rounds` x (Metropolis kernel + GRID_COPY), kernel compilation, random
number generation and host set-up excluded; attempts = P * rounds, rate = attempts / ns. The kernels are compiled as
CUDA-METRO ships them (SourceModule options -std=c++17 -O0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .vampire_pair import flips_per_ns

#: CUDA-METRO main head on 2026-09-05
CUDAMETRO_COMMIT = "cbc9737f18ed7827efea95f3791ff7e99e24dc87"
CUDAMETRO_URL = "https://github.com/arkavo/CUDA-METRO"
PYCUDA_REQ = "pycuda>=2024.1.2"            # requirements.pip at CUDAMETRO_COMMIT
#: J1 in meV (any positive value works: only J1 * beta enters)
J_MEV = 1.0
#: Boltzmann constant in meV/K as construct.py writes it (beta = 1 / (T * 8.6173e-2))
KB_MEV = 8.6173e-2
#: fitted error coefficient of the P-site synchronous update (benchmarks/analyze_bench.py C_ERR)
C_ERR = 0.0276
#: target accuracy for the usable-P ceiling (the stricter of analyze_bench.py TARGETS)
ACCURACY = 0.99
#: bench_speed.py default --blocks and --attempts
P_GRID = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
ATTEMPTS = 2e7
MATERIAL = "SCKHEIS"
RESULT_TAG = "CUDAMETRO_RESULT "
#: lattice -> (Metropolis kernel in montecarlo.py, neighbour-shell signature, kernel takes NVEC)
LATTICES = {"square": ("METROPOLIS_MC_DM0_4_4_4_8", (4, 4, 4, 8), True),
            "triangular": ("METROPOLIS_MC_DM0_6_6_6_12", (6, 6, 6, 12), False)}
NOTE = ("Ours is exact: sites of one colour class share no bond, so each launch equals a sequential single-site "
        "Metropolis sweep over that class (sequential-equivalent). CUDA-METRO's synchronous P-site update is approximate by its own "
        "benchmarks/README.md, deviating from single-update Metropolis by about 0.0276 P/N in the ordered phase. "
        "Its rate counts attempted single-spin updates (P per round), random numbers pre-drawn and untimed; ours "
        "counts attempted updates with in-kernel random numbers. CUDA-METRO is float32 only, so B=1 (float32) is "
        "the equal-precision pairing and 1d (float64) a stricter reference. Kernels compiled as shipped (-O0). "
        "Its site index int(L*L*u) with curand u in (0, 1] can equal L*L; such indices are clamped to L*L - 1 "
        "before timing (clamped_sites per run), the only change to its inputs.")


def _check(lattice: str):
    if lattice not in LATTICES:
        raise ValueError(f"lattice {lattice!r} not paired; supported: {sorted(LATTICES)} (the 3_6_3_6 'honeycomb' "
                         "family uses a non-symmetric neighbour relation, see the module docstring)")


# ------------------------------------------------------------------ units
def kelvin(T: float) -> float:
    """CUDA-METRO temperature (K) of the reduced temperature T (units of J / kB), with J1 = J_MEV meV."""
    return T * J_MEV / KB_MEV


def beta(T: float) -> np.ndarray:
    """The float32 inverse temperature CUDA-METRO writes into BJ (construct.py run_mc_tc_*), in 1/meV."""
    return np.array([1.0 / (np.float32(kelvin(T)) * KB_MEV)], dtype=np.float32)


# ------------------------------------------------------------------ inputs
def material_line(lattice: str) -> str:
    """One row in CUDA-METRO's material format (inputs/TC_*.csv, parsed by read_2dmat with re.split('[,|+]')):
    name, spin, J1..J4, K1x..K4z (12), Ax Ay Az, shell signature a+b+c+d, Tc, DMI."""
    _check(lattice)
    nbs = "+".join(str(k) for k in LATTICES[lattice][1])
    vals = [1.0, J_MEV] + [0.0] * 3 + [0.0] * 12 + [0.0] * 3
    return ",".join([MATERIAL] + [repr(float(v)) for v in vals] + [nbs, "1.0", "0.0"])


def write_material(d: Path, lattice: str) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"TC_{MATERIAL}.csv"
    line = material_line(lattice)
    p.write_text(line + "\n")
    return p


def config_for(lattice: str, size: int, P: int, rounds: int, T: float) -> dict:
    """A CUDA-METRO config (configs/*.json keys, as benchmarks/bench_speed.py config_for builds them), plus the
    lattice. The worker reads SIZE, Blocks, Threads, stability_runs, B, Temps and Material from it."""
    _check(lattice)
    return {"Single_Mat_Flag": 1, "Animation_Flags": 0, "DMI_Flag": 0, "TC_Flag": 1, "Static_T_Flag": 1,
            "FM_Flag": 1, "Input_flag": 0, "Input_File": "Starting_States/state.npy",
            "Temps": [kelvin(T)], "Material": MATERIAL, "Multiple_Materials": "TCS",
            "SIZE": int(size), "Box": 16, "Blocks": int(P), "Threads": 2, "B": "0.0",
            "stability_runs": int(rounds), "calculation_runs": 0, "stability_wrap": 1, "calculation_wrap": 0,
            "Cmpl_Flag": 1, "Prefix": f"sck_{lattice}_{size}_{P}", "Lattice": lattice}


# ------------------------------------------------------------------ rates and P
def metro_rate(P: int, rounds: int, seconds: float) -> float:
    """Attempted single-spin updates per ns (P per round); NaN when the time is not positive."""
    return flips_per_ns(P, rounds, seconds)


def rounds_for(attempts: float, P: int) -> int:
    return max(1, int(attempts) // int(P))


def p_grid(N: int) -> list[int]:
    return [P for P in P_GRID if P <= N]


def accuracy_ceiling(N: int, accuracy: float = ACCURACY) -> int:
    """analyze_bench.py: P <= (1 - A) N / 0.0276."""
    return int(math.floor((1 - accuracy) * N / C_ERR))


def usable_ceiling(N: int, hw: int | None = None, accuracy: float = ACCURACY) -> int:
    """The smaller of the accuracy ceiling and the hardware one (max blocks/SM x SMs), at most N."""
    c = min(accuracy_ceiling(N, accuracy), N)
    return int(min(c, hw)) if hw else c


def pick_fastest(rates: dict) -> int:
    ok = {P: r for P, r in rates.items() if r == r}
    if not ok:
        raise ValueError("no finite CUDA-METRO rate in the P scan")
    return max(ok, key=ok.get)


# ------------------------------------------------------------------ install
def install_cudametro(root: Path) -> Path | None:
    """pip-install pycuda (when missing) and fetch CUDA-METRO at CUDAMETRO_COMMIT into root / "cudametro";
    idempotent. Returns the clone, or None on failure (reported, never hidden)."""
    d = Path(root) / "cudametro"
    try:
        try:
            import pycuda  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", PYCUDA_REQ, "numpy"], check=True)
        head = ""
        if (d / ".git").exists():
            head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        if head != CUDAMETRO_COMMIT:
            d.mkdir(parents=True, exist_ok=True)
            g = ["git", "-C", str(d)]
            subprocess.run(g + ["init", "-q"], check=True)
            subprocess.run(g + ["remote", "remove", "origin"], capture_output=True)
            subprocess.run(g + ["remote", "add", "origin", CUDAMETRO_URL], check=True)
            r = subprocess.run(g + ["fetch", "-q", "--depth", "1", "origin", CUDAMETRO_COMMIT])
            if r.returncode:                         # server refused a fetch by sha: full fetch
                subprocess.run(g + ["fetch", "-q", "origin"], check=True)
            subprocess.run(g + ["checkout", "-q", "--force", CUDAMETRO_COMMIT], check=True)
        if not (d / "src" / "cudametro" / "montecarlo.py").exists():
            raise FileNotFoundError("src/cudametro/montecarlo.py missing after checkout")
    except Exception as e:
        print("CUDA-METRO install failed:", repr(e), flush=True)
        return None
    return d


# ------------------------------------------------------------------ worker (GPU process)
def parse_worker_output(text: str) -> list[dict]:
    recs = [json.loads(line[len(RESULT_TAG):]) for line in text.splitlines() if line.startswith(RESULT_TAG)]
    if not recs:
        raise RuntimeError("CUDA-METRO worker produced no result: " + text[-2000:])
    return recs


def _worker(argv=None):
    """One process: for each config, build CUDA-METRO's buffers as construct.MonteCarlo.__init__, mc_init and
    generate_random_numbers do, then time the kernel loop as benchmarks/bench_speed.py does. Imports montecarlo.py
    and Material_Reader.py directly (the package __init__ imports construct, which needs seaborn and matplotlib)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated config paths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-rounds", type=int, default=50)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.root) / "src" / "cudametro"))
    import pycuda.autoinit  # noqa: F401  (honours CUDA_DEVICE)
    import pycuda.curandom
    import pycuda.driver as drv
    import pycuda.gpuarray as ga
    t0 = time.perf_counter()
    import montecarlo as mc                      # SourceModule compile (cached by pycuda after the first run)
    compile_s = time.perf_counter() - t0
    from Material_Reader import read_2dmat

    dev = drv.Context.get_device()
    attr = drv.device_attribute
    sms = dev.get_attribute(attr.MULTIPROCESSOR_COUNT)
    mbs = dev.get_attribute(attr.MAX_BLOCKS_PER_MULTIPROCESSOR) if hasattr(attr, "MAX_BLOCKS_PER_MULTIPROCESSOR") else None
    rng = np.random.default_rng(a.seed)
    xorwow = pycuda.curandom.XORWOWRandomNumberGenerator()
    for cp in a.configs.split(","):
        cfg = json.loads(Path(cp).read_text())
        lattice, L, P, rounds = cfg["Lattice"], cfg["SIZE"], cfg["Blocks"], cfg["stability_runs"]
        kname, _, nvec = LATTICES[lattice]
        kern = getattr(mc, kname)
        _, params = read_2dmat(str(Path(a.inputs) / f"TC_{cfg['Material']}.csv"))
        params = params.astype(np.float32)
        spin = np.float32(params[0])
        GSIZE = drv.to_device(np.array([L], dtype=np.int32))
        B_GPU = drv.to_device(np.array([np.float32(cfg["B"])], dtype=np.float32))
        SGPU = drv.to_device(np.array([spin], dtype=np.float32))
        GPU_MAT = drv.to_device(params)
        BJ = drv.to_device(np.array([1.0 / (np.float32(cfg["Temps"][0]) * KB_MEV)], dtype=np.float32))
        v = rng.normal(size=(L * L, 3))
        grid0 = (spin * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32).ravel()
        GRID_GPU = drv.to_device(grid0)
        GPU_TRANS = drv.to_device(np.zeros(P * 4, dtype=np.float32))
        NVEC = drv.to_device(np.zeros(18, dtype=np.float32)) if nvec else None   # unused by the dm0 Hamiltonian
        # construct.generate_random_numbers, C1 = Blocks * stability_runs
        C1 = P * rounds
        t1 = time.perf_counter()
        NLIST = xorwow.gen_uniform(C1, np.float32)
        ULIST = xorwow.gen_uniform(C1, np.float32)
        VLIST = xorwow.gen_uniform(C1, np.float32)
        RLIST = xorwow.gen_uniform(C1, np.float32)
        NFULL = ga.zeros(C1, dtype=np.int32)
        S1, S2, S3 = (ga.zeros(C1, dtype=np.float32) for _ in range(3))
        mc.NPREC(NLIST, NFULL, GSIZE, block=(1, 1, 1), grid=(C1, 1, 1))
        # CUDA-METRO's NList_processor maps curand's (0, 1] uniform to __float2uint_rz(L*L*u), so u = 1.0 gives the
        # out-of-range site L*L (an illegal access at L = 1024 on the H100 and A100). Clamp to L*L - 1, untimed,
        # and record how often it happened; this is the only change to its inputs.
        drv.Context.synchronize()
        nf = NFULL.get()
        clamped = int((nf >= L * L).sum())
        if clamped:
            NFULL.set(np.minimum(nf, L * L - 1).astype(np.int32))
        mc.VPREC(ULIST, VLIST, S1, S2, S3, SGPU, block=(1, 1, 1), grid=(C1, 1, 1))
        drv.Context.synchronize()
        rng_s = time.perf_counter() - t1
        del NLIST, ULIST, VLIST

        def step(j):
            sl = slice(j * P, (j + 1) * P)
            args = [GPU_MAT, GRID_GPU, BJ, NFULL[sl], S1[sl], S2[sl], S3[sl], RLIST[sl], GPU_TRANS]
            args += ([NVEC] if nvec else []) + [B_GPU, GSIZE]
            kern(*args, block=(2, 1, 1), grid=(P, 1, 1))
            mc.GRID_COPY(GRID_GPU, GPU_TRANS, block=(2, 1, 1), grid=(P, 1, 1))

        for j in range(min(a.warm_rounds, rounds)):          # untimed: first launches, caches, clocks
            step(j)
        start, end = drv.Event(), drv.Event()
        drv.Context.synchronize()
        start.record()
        for j in range(rounds):
            step(j)
        end.record()
        end.synchronize()
        kernel_s = start.time_till(end) * 1e-3
        g = np.empty_like(grid0)
        drv.memcpy_dtoh(g, GRID_GPU)
        g3 = g.reshape(-1, 3)
        norms = np.linalg.norm(g3, axis=1)
        rec = {"lattice": lattice, "L": L, "N": L * L, "Blocks": P, "rounds": rounds, "attempts": P * rounds,
               "kernel_s": kernel_s, "rng_s": rng_s, "compile_s": compile_s, "clamped_sites": clamped,
               "flips_per_ns": metro_rate(P, rounds, kernel_s),
               "grid_finite": bool(np.isfinite(g).all()),
               "norm_min": float(np.nanmin(norms)), "norm_max": float(np.nanmax(norms)),
               "changed_fraction": float(np.mean(np.any(g3 != grid0.reshape(-1, 3), axis=1))),
               "m_abs": float(np.linalg.norm(np.nanmean(g3, axis=0))),
               "gpu": dev.name(), "sms": sms, "max_blocks_per_sm": mbs,
               "hw_ceiling": (mbs * sms) if mbs else None}
        print(RESULT_TAG + json.dumps(rec), flush=True)
        del NFULL, S1, S2, S3, RLIST, GRID_GPU, GPU_TRANS


def run_worker(root: Path, inputs: Path, configs: list[Path], seed: int, device_index: int = 0,
               timeout: float = 3600) -> list[dict]:
    pkg_parent = str(Path(__file__).resolve().parents[1])
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    env = dict(os.environ, CUDA_DEVICE=str(device_index),
               PYTHONPATH=pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PATH=f"{cuda}/bin" + os.pathsep + os.environ.get("PATH", ""))
    cmd = [sys.executable, "-m", "Spin_Colour_Kernel.cudametro_pair", "--root", str(root), "--inputs", str(inputs),
           "--configs", ",".join(str(c) for c in configs), "--seed", str(seed)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    out = r.stdout + r.stderr
    if r.returncode:
        raise RuntimeError(f"CUDA-METRO worker exited {r.returncode}: " + out[-3000:])
    return parse_worker_output(out)


# ------------------------------------------------------------------ rows
def assemble_row(lattice: str, L: int, n: int, P: int, p_choice: list, metro: list, ours: dict, batches,
                 runs: list | None = None, **extra) -> dict:
    """One JSON row: rates per round and the paired bootstrap of log(ours / CUDA-METRO) per batch."""
    from .exp_perf import boot_ci

    row = {"system": f"Heisenberg {lattice}", "lattice": lattice, "L": L, "active_sites": n, "cudametro_sites": L * L,
           "blocks_P": P, "p_choice": list(p_choice), "P_over_N": P / (L * L),
           "predicted_error": C_ERR * P / (L * L), "metro_flips_per_ns": list(metro),
           "ours_flips_per_ns": {str(B): list(ours[B]) for B in batches}, "cudametro_runs": runs or [],
           "order": "alternated, reversed every round", **extra}
    for B in batches:
        if len(ours[B]) != len(metro):
            raise ValueError(f"unpaired rounds: ours B={B} has {len(ours[B])}, CUDA-METRO {len(metro)}")
        m, lo, hi = boot_ci(np.log(np.asarray(ours[B], float) / np.asarray(metro, float)))
        row[f"log_ratio_B{B}"] = {"mean": m, "low": lo, "high": hi}
    return row


# ------------------------------------------------------------------ E6metro
def cudametro_pair(out, device="cuda:0", reps=10, batches=(1, "1d"), sizes=(256, 512, 1024, 2048),
                   lattices=("square",), warmup=1, tag="E6metro", T: float = 1.0, seed: int = 6007,
                   attempts: float = ATTEMPTS, work: Path = Path("/tmp/sck")):
    """Generated kernel (batch 1 in float32, "1d" = batch 1 in float64) against CUDA-METRO (float32, as shipped) on
    the nearest-neighbour Heisenberg ferromagnet, L x L, reduced temperature T. Per (lattice, L): a P scan over
    bench_speed's grid picks CUDA-METRO's fastest P; the usable ceiling min(accuracy, hardware) is timed too
    (once when equal). Then `warmup` discarded and `reps` measured rounds, sides alternated (order reversed every
    round), each CUDA-METRO round a fresh worker of `attempts` attempts; paired log ratios with bootstrap CIs."""
    import torch

    from . import system as sy
    from .exp_core import save
    from .exp_perf import _events_ms, gpu_state
    from .sampler import Sampler

    dev_index = torch.device(device).index or 0
    root = install_cudametro(work)
    res = {"cudametro_commit": CUDAMETRO_COMMIT, "installed": root is not None, "T_reduced": T, "J1_meV": J_MEV,
           "kB_meV_per_K": KB_MEV, "temperature_K": kelvin(T), "beta_per_meV": float(beta(T)[0]),
           "move": "uniform on the sphere, independent of the old spin, Metropolis",
           "cudametro_precision": "float32", "accuracy_target": ACCURACY, "error_law": f"{C_ERR} P/N",
           "batches": [str(B) for B in batches], "reps": reps, "warmup_rounds_discarded": warmup, "seed": seed,
           "attempts_per_run": attempts, "sizes": list(sizes), "lattices": list(lattices), "note": NOTE, "rows": []}
    if root is None:
        res["status"] = "CUDA-METRO not installed"
        return save(out, tag, res)
    rundir = work / "cudametro-runs"
    for lattice in lattices:
        _check(lattice)
        inputs = rundir / lattice
        write_material(inputs, lattice)
        for L in sizes:
            s = sy.heisenberg(lattice, L)
            n = s.n_active
            N = L * L
            d = inputs / f"L{L}"
            d.mkdir(parents=True, exist_ok=True)

            def cfgs(Ps):
                paths = []
                for P in Ps:
                    p = d / f"cfg_P{P}.json"
                    p.write_text(json.dumps(config_for(lattice, L, P, rounds_for(attempts, P), T), indent=1))
                    paths.append(p)
                return paths

            scan = run_worker(root, inputs, cfgs(p_grid(N)), seed, dev_index)
            rates = {r["Blocks"]: r["flips_per_ns"] for r in scan}
            fastest = pick_fastest(rates)
            hw = scan[0].get("hw_ceiling")
            ceiling = usable_ceiling(N, hw)
            chosen = {fastest: ["fastest"]}
            chosen.setdefault(ceiling, []).append("ceiling")
            Ps = sorted(chosen)

            prep = {}
            for B in batches:
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
            ours = {B: [] for B in batches}
            metro = {P: [] for P in Ps}
            runs = {P: [] for P in Ps}
            paths = cfgs(Ps)
            for i in range(-warmup, reps):       # negative rounds are warm-up, discarded
                def run_ours():
                    for B in batches:
                        replay, upd, dv, nrep = prep[B][:4]
                        v = upd / (_events_ms(replay, nrep, dv) * 1e6)
                        if i >= 0:
                            ours[B].append(v)

                def run_metro():
                    torch.cuda.synchronize(device)
                    for r in run_worker(root, inputs, paths, seed + 1 + i, dev_index):
                        if i >= 0:
                            metro[r["Blocks"]].append(r["flips_per_ns"])
                            runs[r["Blocks"]].append(r)
                for fn in ((run_ours, run_metro) if i % 2 == 0 else (run_metro, run_ours)):
                    fn()
            state = gpu_state(device)
            for P in Ps:
                row = assemble_row(lattice, L, n, P, chosen[P], metro[P], ours, batches, runs=runs[P],
                                   sites_match=(n == N), p_fastest=fastest, p_ceiling=ceiling,
                                   p_accuracy_ceiling=accuracy_ceiling(N), p_hw_ceiling=hw, p_scan=scan,
                                   cudametro_commit=CUDAMETRO_COMMIT, gpu_state=state, note=NOTE)
                res["rows"].append(row)
                for B in batches:
                    lr = row[f"log_ratio_B{B}"]
                    print(f"E6metro {lattice} L={L} P={P} ({'+'.join(chosen[P])}) ours B={B} "
                          f"{np.mean(ours[B]):.3f}/ns  CUDA-METRO {np.mean(metro[P]):.4f}/ns  "
                          f"log ratio {lr['mean']:+.3f} [{lr['low']:+.3f},{lr['high']:+.3f}]", flush=True)
            del prep
            torch.cuda.empty_cache()
    res["status"] = "run"
    return save(out, tag, res)


if __name__ == "__main__":
    _worker()
