"""Command-line runner: python -m Spin_Colour_Kernel.run --exp E1,E2 --out DIR [--device cuda:0] [--smoke]."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="comma list of E1..E8, E4tri, E4trix, E4smoke, Ecompact, Epacked")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes: checks the pipeline, never evidence")
    ap.add_argument("--meas", type=int, default=20000, help="E4: measurement sweeps per chain")
    ap.add_argument("--only", default="", help="E4: comma list of system names to run (e.g. 'Ising sc,XY sc')")
    a = ap.parse_args(argv)
    from . import exp_core as C
    from . import exp_perf as P
    from . import vampire_pair as VP
    from . import boris_pair as BP
    from . import chunk_check as CK
    from . import cudametro_pair as MP
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    sm = a.smoke
    only = [n.strip() for n in a.only.split(",") if n.strip()] or None
    jobs = {
        "E1": lambda: C.e1(out, sweeps=20 if sm else 1000, device=a.device),
        "E2": lambda: C.e2(out, seeds=(1, 2) if sm else tuple(range(1, 11)), samples=5000 if sm else 100_000,
                           batch=200 if sm else 1000, device=a.device),
        # experiments v04/v05, E2 confirmatory: Barker decides H1b, Metropolis is the reported contrast; new seeds
        "E2conf": lambda: (C.e2(out, seeds=(1001, 1002) if sm else tuple(range(1001, 1011)),
                                samples=5000 if sm else 100_000, batch=200 if sm else 1000, device=a.device,
                                acceptance="barker", tag="E2conf"),
                           C.e2(out, seeds=(1001, 1002) if sm else tuple(range(1001, 1011)),
                                samples=5000 if sm else 100_000, batch=200 if sm else 1000, device=a.device,
                                acceptance="metropolis", tag="E2conf-metropolis")),
        "E2b": lambda: C.e2(out, seeds=(1, 2) if sm else tuple(range(1, 11)), samples=5000 if sm else 100_000,
                            batch=200 if sm else 1000, device=a.device, acceptance="barker", tag="E2b"),
        "E3": lambda: C.e3(out),
        "E4": lambda: P.e4(out, device=a.device, only=only, meas=a.meas,
                           tag="E4" if not only else "E4-" + "-".join(n.replace(" ", "_") for n in only)),
        # experiments v19: the Ising critical points again under the Barker rule, whose fixed scan is ergodic
        "E4barker": lambda: P.e4(out, device=a.device, only=only, meas=a.meas, acceptance="barker",
                                 tag="E4barker" if not only else "E4barker-" + "-".join(n.replace(" ", "_") for n in only)),
        "E4tri": lambda: P.e4_triangular_afm(out, device=a.device, sizes=(6, 12) if sm else (24, 48, 96),
                                             meas=2000 if sm else 1_000_000, probe_cap=4000 if sm else 2_000_000),
        # exploratory, outside the declared sizes: does the peak flatten above the crossover once L > xi?
        "E4trix": lambda: P.e4_triangular_afm(out, device=a.device, sizes=(6, 12) if sm else (96, 192),
                                              meas=2000 if sm else 1_000_000, probe_cap=4000 if sm else 2_000_000,
                                              tag="E4-tri-x"),
        # experiments v06, route F: L = 96, 192, 384, more temperatures near the crossover, new seeds
        "E4triconf": lambda: P.e4_triangular_afm(out, device=a.device, sizes=(6, 12) if sm else (96, 192, 384),
                                                 temps=(0.2, 0.25, 0.28, 0.30, 0.32, 0.4),
                                                 meas=2000 if sm else 1_000_000, probe_cap=4000 if sm else 2_000_000,
                                                 tag="E4-tri-conf", seed_base=100_000),
        # experiments v07, route F redesigned: each temperature sampled at the fastest one's interval, new seeds
        "E4triv7": lambda: P.e4_triangular_afm(out, device=a.device, sizes=(6, 12) if sm else (96, 192, 384),
                                               temps=(0.2, 0.25, 0.28, 0.30, 0.32, 0.4),
                                               meas=2000 if sm else 1_000_000, probe_cap=4000 if sm else 2_000_000,
                                               tag="E4-tri-v07", seed_base=200_000, thin="per_T"),
        "Epacked": lambda: P.packing_ab(out, device=a.device, rounds=2 if sm else 10, reps=2 if sm else 5,
                                        sizes=(64, 256) if sm else (256, 1024)),
        "Ecompact": lambda: P.compaction_ab(out, device=a.device, rounds=2 if sm else 10, reps=2 if sm else 5),
        "E4smoke": lambda: P.e4(out, device=a.device, sizes2d=(8, 16, 24), sizes3d=(4, 6, 8), seeds=4,
                                meas=600, boots=200, only=["Ising square", "Heisenberg sc"], tag="E4-smoke"),
        "E6": lambda: P.e6(out, device=a.device, batches=(8, 16) if sm else (8, 16, 32, 64, 128),
                           reps=3 if sm else 10),
        # experiments v04, E6 confirmatory: seeds not used in the v03 runs, the log 1.05 margin in each row
        "E6conf": lambda: P.e6(out, device=a.device, batches=(8, 16) if sm else (8, 16, 32, 64, 128),
                               reps=3 if sm else 10, seed=1004, tag="E6conf"),
        "E1bit": lambda: C.e1_bitwise(out, sweeps=5 if sm else 50, device=a.device),
        "E6bit": lambda: P.e6_bitwise(out, device=a.device, reps=3 if sm else 10),
        # experiments v08: the independent bitwise mode (per-replica draws) and the within-word independence check
        "E1bitind": lambda: C.e1_bitwise(out, sweeps=5 if sm else 50, device=a.device, independent=True, tag="E1bitind"),
        "E6bitind": lambda: P.e6_bitwise(out, device=a.device, reps=3 if sm else 10, independent=True, tag="E6bitind"),
        "Kshape": lambda: P.kshape(out, device=a.device, L=512 if sm else 4096),
        # experiments v12: T4 rerun with 2 discarded warm-up rounds, clocks logged, verdict at the default 10 rounds
        "E6scanv12": lambda: P.e6_scan(out, device=a.device, sides=(4096, 8192) if sm else (4096, 8192, 16384),
                                       reps=2 if sm else 10, warmup=1 if sm else 2, verdict_rounds=10, seed=6012,
                                       tag="E6scan-v12"),
        "E6boris": lambda: BP.boris_pair(out, device=a.device, reps=2 if sm else 10, batches=(1, "1d"),
                                         cells_list=(16,) if sm else (32, 48, 64, 80, 96), warmup=1, crystals=("sc",),
                                         tag="E6boris"),
        "E1chunk": lambda: CK.chunk_check(out, device=a.device, quick=sm),
        "E6metrov17": lambda: MP.cudametro_pair(out, device=a.device, reps=2 if sm else 10, batches=(1, "1d"),
                                                sizes=(256,) if sm else (1024, 2048, 4096),
                                                lattices=("triangular",), warmup=1, tag="E6metro-v17"),
        "E6borisx": lambda: BP.boris_pair(out, device=a.device, reps=2 if sm else 10, batches=(1, "1d"),
                                          cells_list=(24,) if sm else (128, 160, 192), warmup=1, crystals=("sc",),
                                          tag="E6boris-ext"),
        "E6metro": lambda: MP.cudametro_pair(out, device=a.device, reps=2 if sm else 10, batches=(1, "1d"),
                                             sizes=(256,) if sm else (256, 512, 1024, 2048, 4096),
                                             lattices=("triangular",), warmup=1, tag="E6metro"),
        "E6vampv13": lambda: VP.vampire_pair(out, device=a.device, reps=2 if sm else 10, batches=(1, "1d"),
                                             cells_list=(16,) if sm else (64, 80, 96), warmup=1, tag="E6vamp-v13"),
        "E6vampv12": lambda: VP.vampire_pair(out, device=a.device, reps=2 if sm else 4, batches=(1, "1d", 128),
                                             cells_list=(16, 24) if sm else (32, 48, 64), warmup=1, tag="E6vamp-v12"),
        "E6scan": lambda: P.e6_scan(out, device=a.device, sides=(4096, 8192) if sm else (4096, 8192, 16384, 32768),
                                    reps=2 if sm else 10),
        "E6spec": lambda: P.e6_specialists(out, device=a.device, reps=3 if sm else 10),
        # VAMPIRE's CUDA Monte Carlo (uniform move) vs ours at B = 1 and 128, Heisenberg sc / bcc / fcc, T = 1.2 J
        "E6vamp": lambda: VP.vampire_pair(out, device=a.device, reps=2 if sm else 5, batches=(1, "1d", 128)),
        "E6indep": lambda: P.e6_independence(out, device=a.device, L=64 if sm else 512, therm=50 if sm else 2000,
                                             sweeps=40 if sm else 1000),
        "E8": lambda: P.e8(out, reps=3 if sm else 10),
    }
    status = {}
    for e in a.exp.split(","):
        t0 = time.time()
        print(f"=== {e} start", flush=True)
        try:
            if e in ("E5sass", "E5old"):
                import os
                py = sys.executable
                env5 = dict(os.environ, SCK_XLA_EXTRA="")
                if e == "E5sass":
                    subprocess.run([py, "-m", "pip", "install", "-q", "zstandard"])
                if e == "E5old":
                    # no ensurepip on the host: install the pinned JAX into a target directory that shadows the system one
                    tgt = "/tmp/jax0435"
                    subprocess.run([py, "-m", "pip", "install", "-q", "--target", tgt, "jax[cuda12]==0.4.35"],
                                   check=True)
                    env5["PYTHONPATH"] = tgt + os.pathsep + os.environ.get("PYTHONPATH", "")
                    # with a --target install `nvidia.cuda_nvcc` is a namespace package (no __file__): read __path__
                    lib = Path(tgt) / "jax" / "_src" / "lib" / "__init__.py"
                    if lib.exists():
                        lib.write_text(lib.read_text().replace(
                            "pathlib.Path(cuda_nvcc.__file__).parent", "pathlib.Path(list(cuda_nvcc.__path__)[0])"))
                cmd = [py, "-m", "Spin_Colour_Kernel.width_law", "--out", str(out),
                       "--method", "sass" if e == "E5sass" else "dump",
                       "--tag", "E5-sass" if e == "E5sass" else "E5-jax0435"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd, env=env5)
                if r.returncode:
                    raise RuntimeError(f"width_law exited {r.returncode}")
            elif e == "E5prof":
                # Nsight Compute on the E5 baseline step: u = 1 / 4 / 1 grids L1 = 79 / 80 / 81
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.e5_profile", "--out", str(out)] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"e5_profile exited {r.returncode}")
            elif e == "E5profconf":
                # experiments v06, route G: per-kernel test of (11) on new grids L1 = 87-97
                grids = [f"heis:{L}" for L in ((87, 88) if sm else (87, 88, 89, 91, 92, 93, 95, 96, 97))]
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.e5_profile", "--out", str(out), "--grids", *grids]
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"e5_profile exited {r.returncode}")
            elif e in ("E9s1", "E9s124"):
                # experiments v06, E9 (routes D, E): nanodisk Q against temperature and field, one shell set per run
                import json as _json
                fields = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
                J2, J4 = (0.3, -0.1) if e == "E9s124" else (0.0, 0.0)
                cfg = {"params": [[J2, 0.0, J4, 0.177, 0.139, H, 0.88] for H in fields],
                       "names": [f"{e}-H{H:.1f}" for H in fields], "Rd": 6.3 if sm else 18.30, "layers": 3 if sm else 5,
                       "angle": 90.0, "replicas": 16 if sm else 128, "T_start": 20.0, "T_end": 0.1,
                       "n_temps": 4 if sm else 200, "sweeps": 300 if sm else 20000, "every": 10, "seed": 42}
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{e}-config.json").write_text(_json.dumps(cfg, indent=1))
                states = a.only or ",".join(str(i) for i in range(len(fields)))
                from Spin_Colour_Kernel import anneal as AN
                AN.main(["--config", str(out / f"{e}-config.json"), "--states", states, "--out", str(out),
                         "--device", a.device])
            elif e == "Kprof":
                # route A of close-reviewer-gaps: Nsight Compute on the generated kernels
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.e5_profile", "--generated", "--out", str(out)] + (["--quick"] if sm else [])
                if a.only:
                    cmd += ["--grids", *[g.strip() for g in a.only.split(",")]]
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"e5_profile exited {r.returncode}")
            elif e == "E5conf":
                # experiments v04, E5: JAX 0.4.35 / jaxlib 0.4.34 on every device; a host with another JAX
                # gets the pinned one in a target directory that shadows it (as E5old does)
                import os
                from importlib.metadata import version
                env5 = dict(os.environ)
                if version("jax") != "0.4.35":
                    tgt = "/tmp/jax0435"
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", tgt,
                                    "jax[cuda12]==0.4.35", "jaxlib==0.4.34"], check=True)
                    env5["PYTHONPATH"] = tgt + os.pathsep + os.environ.get("PYTHONPATH", "")
                    lib = Path(tgt) / "jax" / "_src" / "lib" / "__init__.py"
                    if lib.exists():
                        lib.write_text(lib.read_text().replace(
                            "pathlib.Path(cuda_nvcc.__file__).parent", "pathlib.Path(list(cuda_nvcc.__path__)[0])"))
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law_b", "--out", str(out), "--tag", "E5conf",
                       "--set", "conf", "--reps", "3" if sm else "10"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd, env=env5)
                if r.returncode:
                    raise RuntimeError(f"width_law_b exited {r.returncode}")
            elif e == "E5genB":
                # experiments v16: the same geometry for B = 1..64, generated kernel against XLA
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law_b", "--out", str(out), "--tag", "E5genB",
                       "--set", "genb", "--reps", "3" if sm else "10"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"width_law_b exited {r.returncode}")
            elif e == "E5pp":
                # periodic against periodic: the E5gen grids moved to even sides, no open-box fallback
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law_b", "--out", str(out), "--tag", "E5pp",
                       "--set", "pp", "--reps", "3" if sm else "10"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"width_law_b exited {r.returncode}")
            elif e == "E5gen":
                # experiments v15, H4 reframed: generated kernel against the XLA baseline (JAX 0.4.35 image)
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law_b", "--out", str(out), "--tag", "E5gen",
                       "--set", "gen", "--reps", "3" if sm else "10"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"width_law_b exited {r.returncode}")
            elif e in ("E5b", "E5c"):
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law_b", "--out", str(out), "--tag", e,
                       "--set", "c" if e == "E5c" else "b", "--reps", "10" if e == "E5c" else "5"] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:
                    raise RuntimeError(f"width_law_b exited {r.returncode}")
            elif e == "E5":
                cmd = [sys.executable, "-m", "Spin_Colour_Kernel.width_law", "--out", str(out)] + (["--quick"] if sm else [])
                r = subprocess.run(cmd)
                if r.returncode:   # an XLA build that rejects the LLVM-IR dump flag: retry without it
                    import os
                    r = subprocess.run(cmd, env=dict(os.environ, SCK_XLA_EXTRA=""))
                if r.returncode:
                    raise RuntimeError(f"width_law exited {r.returncode}")
            else:
                jobs[e]()
            status[e] = {"status": "ok", "seconds": time.time() - t0}
        except Exception as exc:
            status[e] = {"status": "error", "error": repr(exc), "traceback": traceback.format_exc(),
                         "seconds": time.time() - t0}
            print(traceback.format_exc(), flush=True)
        print(f"=== {e} {status[e]['status']} {time.time() - t0:.0f}s", flush=True)
        (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    main()
