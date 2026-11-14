"""Build notebooks/nanodisk_simulation.ipynb: a self-contained cooling run of the magnetic nanodisk on
the generated kernel. The package modules it needs are embedded verbatim with their sha256, so the
notebook runs byte-identical code to this repository without cloning it.

    python notebooks/build_nanodisk_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "Spin_Colour_Kernel"
MODULES = ("__init__.py", "crystals.py", "colouring.py", "system.py", "generator.py", "bitwise.py", "sampler.py",
           "anneal.py")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


INTRO = """
# Nanodisk Monte Carlo on the generated kernel — ready to run

Upload this file to Kaggle or Colab, or run it on your own machine with an NVIDIA GPU. **It does not
need the repository**: it carries the package modules it uses, each checked against its `sha256`, so
it runs code byte-identical to `spin-colour-kernel` at commit `{commit}`.

## What it does

It cools the magnetic nanodisk (sc Heisenberg disc with exchange up to four shells,
bulk-type DMI on the in-plane bonds, cubic and surface anisotropy, Zeeman field) from a hot temperature to 0.1 K, one
state per parameter row, and reports these physical metrics at every
step: E per site, |M|, Mz, Cv, χ and the Berg–Lüscher topological charge Q of the middle layer.

## How to use it

**Edit only the parameters cell.** One row per state; the notebook counts the rows. Then *Run all*.

## Performance

Measured on the paper's benchmarks: on the nanodisk the generated kernel's throughput rises from
5.0 to 13.4 spin flips/ns between 8 and 128 replicas on an H100 (E6), and every state is checked
spin by spin against the sequential chain before it runs.

> Needs a CUDA GPU with compute capability ≥ 7.0 (T4, A100, H100, …). Triton does not run on a P100.
"""

PARAMS = r'''
# ══════════════════════════════════════════════════════════════════════════
#  THE PARAMETERS — the only cell to edit
# ══════════════════════════════════════════════════════════════════════════
#
# One row per state. Each row carries the SEVEN dataset parameters, IN THIS ORDER:
#
#   index  name   what it is                                   units
#   ─────  ─────  ───────────────────────────────────────────  ─────────────
#     0    Jex2   exchange, 2nd sc shell (distance √2)          relative to Jex
#     1    Jex3   exchange, 3rd sc shell (distance √3)          relative to Jex
#     2    Jex4   exchange, 4th sc shell (distance 2)           relative to Jex
#     3    Kan1   cubic anisotropy, first order                 meV
#     4    KanS   surface anisotropy                            meV
#     5    Hex    external field (magnitude)                    meV
#     6    KDM    Dzyaloshinskii–Moriya (bulk type, in-plane)   meV
#
# First-neighbour exchange is Jex = 1 meV, the energy unit. Temperature is not a parameter: every
# state follows the same ladder, so the transition temperature is measured, not declared.
#
# SHELLS AND COLOURINGS. A non-zero Jex2, Jex3 or Jex4 in ANY row switches that shell on for every
# state (one colouring serves the whole run). The colouring is searched and checked on the disk:
#
#   active shells   colours
#   ─────────────   ───────
#   1, 1+3             2
#   1+2                4
#   1+4                3
#   1+3+4              4
#   1+2+4              7
#   1+2+3              8
#   1+2+3+4            9
#
# A sweep is one launch per colour; fewer colours means more parallel work per launch.

PARAMS = [
    # [ Jex2, Jex3, Jex4,  Kan1  , KanS ,  Hex   ,  KDM   ]
    [  0.0,  0.0,  0.0,  0.2867,  0.0 ,  0.7691,  0.6535],  # uniform
    # [  0.0,  0.0,  0.0,  0.1770,  0.139,  0.4830,  0.8800],  # helical
    # [  0.0,  0.0,  0.0,  0.3500,  0.0 ,  0.2000,  0.9500],  # points
    # [  0.0,  0.0,  0.0,  0.1000,  0.200,  0.9000,  0.3000],  # lines
]
NAMES = ["uniform"]          # one name per row; left short, the rest are numbered

# ══════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════════════
RD     = 18.30   # disk radius in lattice steps; the box is N = 2(⌊Rd⌋+1)+1 wide
LAYERS = 5       # layers in z
ANGLE  = 90.0    # field direction in degrees, in the x–z plane

# ══════════════════════════════════════════════════════════════════════════
#  SCHEDULE
# ══════════════════════════════════════════════════════════════════════════
REPLICAS      = 128     # independent chains PER STATE. More is cheaper per update: no penalty
T_START       = 20.0    # K
T_END         = 0.1     # K
N_TEMPS       = 200     # steps: 200 gives exactly 0.1 K steps
SWEEPS        = 20000   # sweeps per temperature (2/3 thermalize, 1/3 measure)
MEASURE_EVERY = 10      # read the observables every this many sweeps of the measuring third
SEED          = 42

# ══════════════════════════════════════════════════════════════════════════
#  GPUS
# ══════════════════════════════════════════════════════════════════════════
GPUS = "all"            # "all", or a list such as [0] or [0, 1]
'''

ENV = r'''
# Environment: the GPU, and the two libraries the kernel needs (PyTorch with CUDA, and Triton).
import subprocess, sys

def _sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""

print("GPU:", _sh("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                  "--format=csv,noheader") or "none detected")
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA GPU visible: this notebook needs one (Kaggle: Settings > Accelerator)")
try:
    import triton
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "triton"], check=True)
    import triton
caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
if any(c < (7, 0) for c in caps):
    raise SystemExit(f"Triton needs compute capability >= 7.0; found {caps}")
print(f"torch {torch.__version__}  triton {triton.__version__}  "
      f"{torch.cuda.device_count()} GPU(s), capability {caps}")
'''

READ = r'''
# Read the parameters, count the states, and build one state to show the colouring the run will use.
import numpy as np
import importlib, sys
sys.path.insert(0, ".")
import Spin_Colour_Kernel.anneal as AN

P = np.asarray(PARAMS, dtype=float)
if P.ndim != 2 or P.shape[1] != 7:
    raise SystemExit(f"PARAMS must be a list of rows of 7 numbers; got shape {P.shape}")
N_STATES = P.shape[0]
names = (list(NAMES) + [f"state-{i+1}" for i in range(len(NAMES), N_STATES)])[:N_STATES]
if len(set(names)) != N_STATES:
    raise SystemExit("NAMES must be distinct: each state is written to its own file")

print(f"{N_STATES} state(s)")
print(f"{'':>3} {'state':<12}" + "".join(f"{n:>8}" for n in AN.PARAM_NAMES))
for i, nm in enumerate(names):
    print(f"{i+1:>3} {nm:<12}" + "".join(f"{v:>8.4f}" for v in P[i]))

shells = AN.active_shells(P)
probe = AN.build_state(P[0], RD, LAYERS, ANGLE, shells)
print(f"\nactive shells: {shells}")
print(f"colouring: {probe.colouring.describe()}  ->  {probe.colouring.n_colours} colours, "
      f"checked admissible on the disk")
print(f"disk: N = {probe.box[0]}, {LAYERS} layers, {probe.n_active} active sites")

n_therm = SWEEPS * 2 // 3
print(f"\nwork: {N_STATES} states x {REPLICAS} replicas, {N_TEMPS} temperatures x {SWEEPS} sweeps "
      f"({n_therm} thermalize, {SWEEPS - n_therm} measure, read every {MEASURE_EVERY})")
print(f"      {N_STATES * REPLICAS * N_TEMPS * SWEEPS * probe.n_active:.3e} spin updates in all")
'''

RUN = r'''
# Run: one process per GPU, states dealt round-robin. Each process first checks every state spin by
# spin against the sequential reference chain (3 sweeps, 2 replicas) and refuses to run a state that
# differs; then it runs the ladder and writes state_<name>.npz.
import json, os, threading, time

devs = list(range(torch.cuda.device_count())) if GPUS == "all" else list(GPUS)
devs = devs[:N_STATES] or [0]
cfg = dict(params=P.tolist(), names=names, Rd=RD, layers=LAYERS, angle=ANGLE, T_start=T_START,
           T_end=T_END, n_temps=N_TEMPS, sweeps=SWEEPS, replicas=REPLICAS, every=MEASURE_EVERY, seed=SEED)
with open("run.json", "w") as f:
    json.dump(cfg, f, indent=1)
os.makedirs("results", exist_ok=True)

procs = []
for j, g in enumerate(devs):
    mine = [i for i in range(N_STATES) if i % len(devs) == j]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=os.getcwd())
    cmd = [sys.executable, "-u", "-m", "Spin_Colour_Kernel.anneal", "--config", "run.json",
           "--states", ",".join(map(str, mine)), "--out", "results", "--device", "cuda:0"]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs.append((g, mine, p))
    print(f"GPU {g}: states {[names[i] for i in mine]}")

def _pump(g, p):
    for line in p.stdout:
        print(f"GPU {g} {line}", end="", flush=True)

t0 = time.perf_counter()
th = [threading.Thread(target=_pump, args=(g, p)) for g, _, p in procs]
for t in th: t.start()
for t in th: t.join()
bad = [(g, p.wait()) for g, _, p in procs if p.wait() != 0]
print(f"\ntotal: {(time.perf_counter() - t0) / 60:.2f} min")
if bad:
    raise SystemExit(f"failed on GPU(s) {bad}: read the lines above")
'''

METRICS = r'''
# The physical metrics:
#
#   E/site  total energy over active sites, each bond counted once
#   |M|     norm of the mean magnetization over active sites; Mz its z component
#   Cv      <(ΔE)²> / (k_B T² N)       from fluctuations about a per-temperature reference (float64)
#   χ       <(Δ|M|)²> / (k_B T)
#   Q       Berg–Lüscher charge of the middle layer (quantised)
import json
out, transitions, Mvec = {}, {}, {}
for nm in names:
    z = np.load(f"results/state_{nm}.npz")
    red = AN.reduce_obs(z["obs"], z["temperatures"], int(z["n_active"]))
    out[nm] = {k: np.asarray(v).tolist() for k, v in red.items()}
    out[nm].update(seconds=float(z["seconds"]), flips_per_ns=float(z["flips_per_ns"]),
                   colours=int(z["colours"]), _snap=z["snaps"][-1].tolist())
    Mvec[nm] = z["M"]

print(f"{'state':<12}{'T':>7}{'E/site':>11}{'|M|':>9}{'Mz':>9}{'Cv':>10}{'χ':>10}{'Q':>9}"
      f"{'flips/ns':>10}")
for nm in names:
    r = out[nm]
    print(f"{nm:<12}{r['T'][-1]:>7.2f}{r['E'][-1]:>11.5f}{r['M'][-1]:>9.4f}{r['Mz'][-1]:>9.4f}"
          f"{r['Cv'][-1]:>10.4f}{r['chi'][-1]:>10.4f}{r['Q'][-1]:>9.4f}{r['flips_per_ns']:>10.2f}")

# The transition, located at the peak of each response function on the 0.1 K grid (no interpolation:
# a parabola through three noisy points claims a precision the data do not have). A finite disk shows
# a rounded maximum, not a divergence; a peak at either end of the ladder means the transition is
# outside the range and the number means nothing.
print(f"\n{'state':<12}{'T(Cv peak)':>12}{'Cv max':>10}{'T(χ peak)':>12}{'χ max':>10}  note")
for nm in names:
    r = out[nm]
    T = np.asarray(r["T"])
    row, msg = {}, []
    for k in ("Cv", "chi"):
        v = np.where(np.isfinite(r[k]), r[k], -np.inf)
        j = int(np.argmax(v))
        row[f"T_{k}"], row[f"{k}_max"] = float(T[j]), float(v[j])
        if j in (0, len(T) - 1):
            msg.append(f"{k} peak at the edge")
    transitions[nm] = row
    print(f"{nm:<12}{row['T_Cv']:>12.1f}{row['Cv_max']:>10.4f}{row['T_chi']:>12.1f}{row['chi_max']:>10.4f}"
          f"  {'; '.join(msg) or 'ok'}")

with open("results.json", "w") as f:
    json.dump(dict(params=P.tolist(), names=names, config=cfg, shells=shells,
                   colouring=probe.colouring.describe(), n_active=probe.n_active,
                   results=out, transitions=transitions), f, indent=1)
print("\nsaved results.json")
print("dataset: results/state_<name>.npz, one per state, each with")
print("  params (7,)  M (replicas, temperatures, 3)  temperatures (n_T,)  obs (n_T, 8, replicas)  snaps")
for nm in names:
    print(f"  results/state_{nm}.npz   M {Mvec[nm].shape} [replica, temperature, component]")
'''

FIGS = r'''
# Cooling curves per state, and the final middle-layer s_z of replica 0.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for nm in names:
    r = out[nm]
    for a, k, t in ((ax[0, 0], "E", "energy per site"), (ax[0, 1], "M", "|M|"),
                    (ax[1, 0], "Cv", "Cv"), (ax[1, 1], "chi", "χ")):
        a.plot(r["T"], r[k], lw=1.4, label=nm)
        a.set_xlabel("T (K)"); a.set_ylabel(t); a.grid(alpha=.25); a.invert_xaxis()
ax[0, 0].legend(fontsize=9)
fig.suptitle(f"{N_STATES} state(s) · {REPLICAS} replicas · {N_TEMPS}×{SWEEPS} sweeps", y=1.0)
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, N_STATES, figsize=(3.1 * N_STATES, 3.3))
axs = np.atleast_1d(axs)
for a, nm in zip(axs, names):
    im = a.imshow(np.asarray(out[nm]["_snap"]), cmap="jet", vmin=-1, vmax=1, origin="lower")
    a.set_title(f"{nm}  ($s_z$)", fontsize=10); a.axis("off")
fig.colorbar(im, ax=axs, fraction=.025)
plt.show()
'''


def module_cell(name: str) -> dict:
    src = (PKG / name).read_text()
    h = hashlib.sha256(src.encode()).hexdigest()
    return code(f'''# Spin_Colour_Kernel/{name} — byte-identical to the repository. sha256 {h}
import hashlib, os
_src = {src!r}
assert hashlib.sha256(_src.encode()).hexdigest() == "{h}", "{name}: sha256 mismatch"
os.makedirs("Spin_Colour_Kernel", exist_ok=True)
with open("Spin_Colour_Kernel/{name}", "w") as f:
    f.write(_src)
print("wrote Spin_Colour_Kernel/{name}  sha256 {h[:16]}…")
''')


def build() -> dict:
    cells = [md(INTRO.replace("{commit}", commit())), md("## 1 · Parameters\n\nThe only cell to edit."),
             code(PARAMS), md("## 2 · Environment"), code(ENV),
             md("## 3 · The modules\n\nWritten to disk from the literals below, each checked against its `sha256`.")]
    cells += [module_cell(m) for m in MODULES]
    cells += [md("## 4 · Reading and checking the parameters"), code(READ),
              md("## 5 · The run"), code(RUN), md("## 6 · Physical metrics and the dataset"), code(METRICS),
              md("## 7 · Figures"), code(FIGS)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = ROOT / "notebooks" / "nanodisk_simulation.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", out)
