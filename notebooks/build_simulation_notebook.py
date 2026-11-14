"""Build notebooks/spin_simulation.ipynb: a self-contained Monte Carlo run of any Hamiltonian the package
supports (Ising, XY or Heisenberg; exchange per shell, DMI, cubic and surface anisotropy, field), on any
crystal of the package, in a box with an optional shape of active sites. It runs on every GPU it finds,
or on the CPU when there is none. The package modules it needs are embedded verbatim with their sha256.

    python notebooks/build_simulation_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nanodisk_notebook import ROOT, code, commit, md, module_cell  # noqa: E402

MODULES = ("__init__.py", "crystals.py", "colouring.py", "system.py", "generator.py", "bitwise.py", "sampler.py",
           "simulate.py")

INTRO = """
# Spin Monte Carlo on the generated multi-colour kernel — ready to run

Upload this file to Kaggle or Colab, or open it in Jupyter on your own machine, and press **Run all**.
**You do not need to know Python, GPUs or the repository**: the notebook carries the package modules it
uses, each checked against its `sha256`, so it runs code byte-identical to `spin-colour-kernel` at
commit `{commit}`.

## What it does

It cools every *state* you define (one set θ of Hamiltonian parameters) from `T_MAX` down to `T_MIN`,
with many independent replicas per state, and records at every temperature the energy, the
magnetization vector, |M| and M² (so the specific heat Cv and the susceptibility χ follow), and the full
spin configuration of every replica.

## How to use it

1. **Edit only cell 1 (the parameters).** Every setting is explained next to it.
2. *Run all*. The notebook finds your hardware by itself: every NVIDIA GPU it sees (one process per GPU,
   states dealt round-robin), or the CPU when there is none.
3. Read the plots (E, |M|, Cv, χ against T) and download the output folder.

> GPU: an NVIDIA GPU with compute capability ≥ 7.0 (T4, A100, H100, …; Triton does not run on a P100).
> CPU: Linux only (Colab, Kaggle without accelerator, WSL), through the Triton interpreter. It is
> correct but **hundreds of times slower**: keep the box, the replicas and the sweeps small there.
"""

PARAMS = r'''
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  1 · THE PARAMETERS — the only cell to edit
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# ── 1.1  TYPE OF HAMILTONIAN ────────────────────────────────────────────────────────────────────
#   "ising"       spins s = ±1                   (1 component)
#   "xy"          unit spins in the plane         (2 components)
#   "heisenberg"  unit spins in 3D                (3 components)
MODEL = "heisenberg"

# ── 1.2  THE HAMILTONIAN (how it is written) ────────────────────────────────────────────────────
#
#   H = − Σ_s J_s Σ_<ij> in shell s  S_i · S_j    exchange, one constant per neighbour shell (each pair once)
#       − Σ_<ij>  S_i^T A_ij S_j      bulk-type (Bloch) DMI on in-plane nearest-neighbour bonds, A_lm = D Σ_k r̂_k ε_klm
#       + Σ_i K1_i (sx² sy² + sx² sz² + sy² sz²)  +  K2 Σ_i (sx sy sz)²      cubic anisotropy
#       − Σ_i  H · S_i                                           Zeeman field
#
#   J_s > 0 is ferromagnetic, J_s < 0 antiferromagnetic. Shell 1 = nearest neighbours, 2 = next
#   nearest, and so on (shells 1–4 on sc). K1_i = Ks on SURFACE sites (an active site with fewer active
#   nearest neighbours than a bulk site), K1 elsewhere.
#   Field direction: Heisenberg (sinθ cosφ, sinθ sinφ, cosθ) with θ = H_theta, φ = H_phi in degrees
#   (θ = 0 is +z, the normal of a film or disc); XY (cosφ, sinφ); Ising +1.
#   D, K1, Ks and K2 exist only for "heisenberg"; Ising and XY carry J and H.
#
# ── 1.3  THE STATES: one set θ of parameters each — add as many as you want ─────────────────────
#   name     a label for the plots and files
#   J        {shell: constant}, e.g. {1: 1.0} or {1: 1.0, 2: -0.2}
#   D        Dzyaloshinskii–Moriya constant (bulk type: its vector lies along the bond)
#   K1, Ks   cubic anisotropy, bulk and surface (Ks left out = K1)
#   K2       second-order cubic anisotropy
#   H, H_theta, H_phi   field magnitude and direction (degrees)
#   A parameter left out is zero.
STATES = [
    dict(name="ferromagnet", J={1: 1.0}),
    dict(name="with field",  J={1: 1.0}, H=0.2, H_theta=0.0),
    # dict(name="nanodisk", J={1: 1.0}, D=0.88, K1=0.177, Ks=0.139, K2=0.034, H=0.483, H_theta=0.0),
]

# ── 1.4  UNITS ──────────────────────────────────────────────────────────────────────────────────
#   "reduced"  energies in units of your J, k_B = 1 (T is in the same units)
#   "meV"      energies in meV, T in kelvin (k_B = 0.086173404 meV/K)
UNITS = "reduced"

# ── 1.5  CRYSTAL STRUCTURE AND BOX ──────────────────────────────────────────────────────────────
#   2D: "square", "triangular", "honeycomb", "kagome"      BOX = [Lx, Ly]
#   3D: "sc", "bcc", "fcc" (bcc, fcc: conventional cubic cells of 2 and 4 sites)   BOX = [Lx, Ly, Lz]
#   The box is counted in unit cells; a square box has equal sizes, a rectangular one does not.
#   PERIODIC: one True/False per axis. If no colouring fits a periodic box, it falls back to open.
CRYSTAL  = "sc"
BOX      = [12, 12, 12]
PERIODIC = [True, True, True]

# ── 1.6  ACTIVE SITES (a shape cut out of the box) ──────────────────────────────────────────────
#   First the coordinate system, then the shape written in it. The origin is the centre of the box;
#   lengths are in lattice units; angles in radians.
#     "cartesian"    x, y, z
#     "spherical"    r, theta, phi        a sphere of radius 5:            SHAPE = "r <= 5"
#     "cylindrical"  rho, phi, z          a disc of radius 5, 3 layers:    SHAPE = "(rho <= 5) & (abs(z) <= 1)"
#   Combine conditions with & (and), | (or), ~ (not). Leave BOTH empty to keep every site active.
#   Sites outside the shape carry no spin; a shape usually goes with PERIODIC = [False, False, False].
COORDS = ""
SHAPE  = ""

# ── 1.7  TEMPERATURES AND SWEEPS ────────────────────────────────────────────────────────────────
T_MAX  = 2.0          # first (hottest) temperature
T_MIN  = 0.2          # last (coldest) temperature
T_STEP = 0.1          # step of the ladder
THERMALIZATION = 1000 # sweeps at each temperature before measuring
MEASUREMENT    = 2000 # sweeps at each temperature while measuring
MEASURE_EVERY  = 10   # read E and M every this many measuring sweeps
REPLICAS       = 32   # independent chains per state (more is cheaper per spin update on a GPU)
SEED           = 1

# ── 1.8  OUTPUT ─────────────────────────────────────────────────────────────────────────────────
#   CONFIG_EVERY = 1 saves every replica's full configuration at every temperature step;
#   2 at every second step, and so on.
CONFIG_EVERY = 1
OUTPUT_DIR   = "simulation_output"
GPUS         = "all"   # "all", or a list such as [0] or [0, 1]; ignored on CPU
'''

ENV = r'''
# 2 · Hardware: every usable NVIDIA GPU, or the CPU through the Triton interpreter.
import os, subprocess, sys

def _sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""

import torch
caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())] \
    if torch.cuda.is_available() else []
usable = [i for i, c in enumerate(caps) if c >= (7, 0)]
if caps and not usable:
    print(f"GPU(s) found with capability {caps}, below the 7.0 Triton needs: using the CPU")
if GPUS != "all":
    usable = [g for g in usable if g in list(GPUS)]
DEVICE = "cuda" if usable else "cpu"
if DEVICE == "cpu":
    os.environ["TRITON_INTERPRET"] = "1"      # must precede the first import of triton
try:
    import triton
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "triton"], check=True)
    import triton
if DEVICE == "cuda":
    print("GPU:", _sh("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"))
    print(f"using {len(usable)} GPU(s): {usable}")
else:
    print("no usable GPU: running on the CPU (Triton interpreter, slow; keep the run small)")
print(f"torch {torch.__version__}  triton {triton.__version__}")
'''

PREPARE = r'''
# 4 · Read and check the parameters, build the first state, and show the size of the output.
import json
import numpy as np
sys.path.insert(0, os.getcwd())
import Spin_Colour_Kernel.simulate as SIM

cfg = dict(model=MODEL, crystal=CRYSTAL, box=list(BOX), periodic=list(PERIODIC), coords=COORDS, shape=SHAPE,
           states=[dict(s) for s in STATES], units=UNITS, T_max=T_MAX, T_min=T_MIN, T_step=T_STEP,
           thermalization=THERMALIZATION, measurement=MEASUREMENT, measure_every=MEASURE_EVERY,
           replicas=REPLICAS, config_every=CONFIG_EVERY, seed=SEED)
for i, s in enumerate(cfg["states"]):
    s.setdefault("name", f"state-{i + 1}")
    s["J"] = {str(k): float(v) for k, v in (s.get("J") or {1: 1.0}).items()}
names = [s["name"] for s in cfg["states"]]
if len(set(names)) != len(names):
    raise SystemExit("state names must be distinct")
RUN = SIM.prepare(cfg, OUTPUT_DIR)
N_STATES = len(names)
s0 = SIM.build_system(MODEL, CRYSTAL, BOX, RUN["periodic_used"], RUN["states"][0], np.load(f"{OUTPUT_DIR}/mask.npy"))
print(f"colouring: {s0.colouring.n_colours} colours, checked admissible on the box; "
      f"periodic {RUN['periodic_used']}")
print(f"temperatures: {len(RUN['temperatures'])} from {RUN['temperatures'][0]} to {RUN['temperatures'][-1]}")
print(f"configurations saved at {RUN['n_saved']} temperature(s) per state "
      "(the run goes on whatever the size; make sure the disk can hold it)")
'''

RUN_CELL = r'''
# 5 · The run: one process per GPU with the states dealt round-robin, or one process on the CPU.
import threading, time

t0 = time.perf_counter()
if DEVICE == "cpu":
    SIM.run(RUN, list(range(N_STATES)), OUTPUT_DIR, device="cpu", log=print)
else:
    devs = usable[:N_STATES]
    procs = []
    for j, g in enumerate(devs):
        mine = [i for i in range(N_STATES) if i % len(devs) == j]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=os.getcwd())
        cmd = [sys.executable, "-u", "-m", "Spin_Colour_Kernel.simulate", "--run", OUTPUT_DIR,
               "--states", ",".join(map(str, mine)), "--device", "cuda:0"]
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True)))
        print(f"GPU {g}: states {[names[i] for i in mine]}")

    def _pump(g, p):
        for line in p.stdout:
            print(f"GPU {g} {line}", end="", flush=True)

    th = [threading.Thread(target=_pump, args=gp) for gp in procs]
    for t in th: t.start()
    for t in th: t.join()
    bad = [g for g, p in procs if p.wait() != 0]
    if bad:
        raise SystemExit(f"failed on GPU(s) {bad}: read the lines above")
print(f"\ntotal: {(time.perf_counter() - t0) / 60:.2f} min")
'''

COLLECT = r'''
# 6 · The results, combined over states, and what each file holds.
res = SIM.collect(OUTPUT_DIR)
T = res["temperatures"]
print(f"{'state':<16}{'T':>9}{'E/site':>11}{'|M|':>9}{'Cv':>10}{'χ':>10}")
for i, nm in enumerate(names):
    print(f"{nm:<16}{T[-1]:>9.3f}{res['E_site'][i, -1]:>11.5f}{res['M_site'][i, -1]:>9.4f}"
          f"{res['Cv'][i, -1]:>10.4f}{res['chi'][i, -1]:>10.4f}")
print(f"""
{OUTPUT_DIR}/
  magnetization.npy    {res['magnetization'].shape}  [state, replica, temperature, component]  <M> per site
  configurations.npy   {tuple(np.load(f"{OUTPUT_DIR}/configurations.npy", mmap_mode="r").shape)}  [state, replica, saved T, site, component]
  sites.npy            positions of the active sites, the 'site' axis above
  results.npz          temperatures, energy and energy2 (<E>, <E²> per replica, totals), Mabs (<|M|>), M2 (<|M|²>),
                       Cv = (<E²>−<E>²)/(k_B T² N) and χ = N(<|M|²>−<|M|>²)/(k_B T), replica means [state, T]
  run.json             every setting of this run, the ladder and the saved temperatures
""")
'''

FIGS = r'''
# 7 · Energy, magnetization, specific heat and susceptibility against temperature.
import matplotlib.pyplot as plt

unit = "K" if UNITS.lower() == "mev" else "J/k_B"
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for i, nm in enumerate(names):
    for a, k, t in ((ax[0, 0], "E_site", "energy per site"), (ax[0, 1], "M_site", "|M| per site"),
                    (ax[1, 0], "Cv", "Cv per site"), (ax[1, 1], "chi", "χ")):
        a.plot(T, res[k][i], "o-", ms=3, lw=1.2, label=nm)
        a.set_xlabel(f"T ({unit})"); a.set_ylabel(t); a.grid(alpha=.25)
ax[0, 0].legend(fontsize=9)
fig.suptitle(f"{MODEL} on {CRYSTAL} {list(BOX)} · {N_STATES} state(s) · {REPLICAS} replicas", y=1.0)
plt.tight_layout(); plt.show()
'''


def build() -> dict:
    cells = [md(INTRO.replace("{commit}", commit())), md("## 1 · Parameters\n\nThe only cell to edit."),
             code(PARAMS), md("## 2 · Hardware"), code(ENV),
             md("## 3 · The modules\n\nWritten to disk from the literals below, each checked against its `sha256`.")]
    cells += [module_cell(m) for m in MODULES]
    cells += [md("## 4 · Reading and checking the parameters"), code(PREPARE),
              md("## 5 · The run"), code(RUN_CELL), md("## 6 · Results and files"), code(COLLECT),
              md("## 7 · Figures"), code(FIGS)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = ROOT / "notebooks" / "spin_simulation.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", out)
