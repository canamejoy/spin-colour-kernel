# spin-colour-kernel

Exact, replica-batched, multi-colour Metropolis Monte Carlo for classical O(n) spin models (Ising, XY,
Heisenberg) on crystals with a basis, with longer-range exchange shells, masks, fields and anisotropy. One
generator turns a *system* (spin model, crystal, finite geometry, couplings) into one specialized
[Triton](https://github.com/triton-lang/triton) GPU kernel; changing the system changes data, not code.

Updates are exact because sites are grouped by a **linear colouring** `gamma(c, alpha) = K c + o_alpha (mod m)`
that no bond can join to itself, so every colour class is updated in parallel and each launch equals a
sequential single-site Metropolis sweep over that class. The colouring is searched in increasing colour count
and certified against the clique number of the bond graph.

## Program summary

| Field | Value |
|---|---|
| Program title | spin-colour-kernel |
| Developer's repository link | https://github.com/canamejoy/spin-colour-kernel |
| Licensing provisions | MIT |
| Programming language | Python 3 (Triton GPU kernels, PyTorch tensors) |
| Hardware | NVIDIA GPU with CUDA 12 (tested on T4, A100 40/80 GB, H100 80 GB); CPU for the test suite (Triton interpreter) |
| Nature of problem | Equilibrium sampling of classical spin lattices needs many independent replicas and long chains; GPU codes are either specialized to one model and lattice or general but slow, and array-compiler implementations depend on compiler decisions about the grid shape. |
| Solution method | Linear colourings of the bond graph (exact parallel classes); a generator that emits one Triton kernel per system, batched over replicas, with counter-based Philox random numbers so that the parallel sweep reproduces a sequential reference chain bit for bit; an optional bitwise multi-spin-coded Ising mode (32 replicas per word, independent per-replica draws). |
| Restrictions | With linear colourings a periodic side that is prime needs as many colours as its length; the bitwise mode applies to Ising with batches that fill 32-replica words. |

## Simulating your own Hamiltonian

`ready_for_run/` holds the notebook with a step-by-step user manual in English and Spanish (Markdown and PDF),
and `configurator.html`, an offline page that shows the Hamiltonian, the lattice, the active sites and the
colouring as you set them, and downloads the notebook or a `.py` script ready to run
(`python notebooks/build_configurator.py` rebuilds it).
`notebooks/spin_simulation.ipynb` is a self-contained notebook for users who do not program: upload it to
Kaggle or Colab (or open it in Jupyter), edit its single parameters cell and press *Run all*. That cell sets
the model (Ising, XY, Heisenberg), the states (one parameter set each: exchange per shell, DMI, cubic and
surface anisotropy, field and its direction), the units, the crystal and box, the active sites as a shape
written in Cartesian, spherical or cylindrical coordinates, the temperature ladder, the sweeps and the
replicas. The notebook uses every NVIDIA GPU it finds (one process per GPU) or falls back to the CPU through
the Triton interpreter, and writes `magnetization.npy` (states, replicas, temperatures, 3),
`configurations.npy` (states, replicas, saved temperatures, sites, 3), `results.npz` (energy, <E^2>, |M|,
M^2, Cv, chi) and `run.json`, with plots of E, |M|, Cv and chi. It embeds the package modules with their
sha256; `python notebooks/build_simulation_notebook.py` regenerates it, and `python notebooks/build_manuals.py`
rebuilds the manuals and copies the notebook into `ready_for_run/`.

## Connecting an agent (MCP)

**From a clone, with no installation:** open the repository in Claude Code and approve the `sck` server it
declares in `.mcp.json` (it runs `python3 sck_mcp.py`; describing, checking and writing runs needs only numpy,
and if numpy is missing the server says how to install it). `AGENTS.md` tells any agent how to use the tools.

`python -m Spin_Colour_Kernel.mcp` is a Model Context Protocol server (stdio, standard library only) through
which an AI agent prepares, checks, writes, launches and reads runs. Register it in your agent, for example
`claude mcp add sck -- python -m Spin_Colour_Kernel.mcp`. Its tools:

| Tool | What it does |
|---|---|
| `describe` | The Hamiltonian, models, crystals, coordinate systems, state parameters, units and every default |
| `check` | Validates a configuration without running it: errors, active and surface sites, colouring per state, temperatures, run size |
| `write` | Writes the run as a ready-to-run `.ipynb` or `.py`, the same file the configurator page downloads |
| `backends` | Which of local, Kaggle, Modal, RunPod and Colab this machine can use, and what to install or configure otherwise |
| `run` | Without `confirm: true`, returns a plan (GPU, cost drivers, time limit, where results land); with it, submits |
| `status`, `results`, `cancel` | Follow a job, fetch and summarize its results (temperatures, \|M\| and energy per site per state, `summary.png`), stop it |

Backends use the user's own access and are detected by presence only; no credential value is ever read or
returned. Kaggle needs the `kaggle` CLI and its token, Modal the `modal` CLI and its token, RunPod the
`RUNPOD_API_KEY` environment variable (the pod installs this package from GitHub at the release tag, then
serves the results behind a per-job token until `results` downloads them and terminates the pod; the pod
bills at its GPU rate from creation until `results` or `cancel` terminates it). Colab has no
submission API here: `run` writes the notebook and points to Google's Colab MCP server for opening it. Jobs live
in `./sck_jobs` (or `SCK_JOBS_DIR`).

## Installation

```bash
pip install spin-colour-kernel
```

On Colab and Kaggle torch and triton are already there, so this adds only the package. Then, in a notebook:

```python
import sck                                    # the short name of Spin_Colour_Kernel

sck.check(crystal="sc", box=[16, 16, 16])     # sites, colouring and run size, without running
res = sck.run(crystal="sc", box=[16, 16, 16],
              states=[dict(name="ferromagnet", J={1: 1.0})],
              T_max=2.0, T_min=0.5, T_step=0.1, replicas=32)
res.plot()                                    # E, |M|, Cv and chi against T
```

The Hamiltonian is chosen by `model` (`"ising"`, `"xy"` or `"heisenberg"`) and `states`, one dict per
parameter set: `J={shell: value}` (shells 1 to 4), `D` (bulk DMI), `K1`, `Ks`, `K2` (cubic, surface and
second-order anisotropy) and `H`, `H_theta`, `H_phi` (field and its direction in degrees). A parameter left
out is zero and `Ks` defaults to `K1`; D and the anisotropies apply to Heisenberg only. The other settings
(`crystal`, `box`, `periodic`, `coords`, `shape`, `units`, the temperature ladder, sweeps and replicas) are the
notebook's parameters in lower case. `examples/library_nanodisk.ipynb` runs a production nanodisk this way and
draws sz of its middle layer while it cools; the manuals in `ready_for_run/` explain every setting.

`sck.describe()` lists every setting and its default; `sck.run` uses every NVIDIA GPU it finds (one process
per GPU) or the CPU through the Triton interpreter, and writes the same output folder as the notebook;
`sck.load(folder)` reads a finished run again; `sck.write("run.ipynb", ...)` writes the ready-to-run
notebook. triton is installed on Linux only: elsewhere the package describes, checks and writes runs, and
`sck-mcp` starts the MCP server for agents.

From the repository, for development:

```bash
git clone https://github.com/canamejoy/spin-colour-kernel
cd spin-colour-kernel
pip install -e ".[dev]"
```

The GPU runs of the paper used Python 3.11, `torch==2.5.1` (CUDA 12.4) and `triton==3.1.0`. The CPU test suite
also passes with recent CPU-only torch and triton through the Triton interpreter.

## Test run

```bash
pytest                                                    # 179 tests, CPU only (Triton interpreter), ~25 min
TRITON_INTERPRET=1 python examples/quickstart.py --device cpu
python examples/quickstart.py --device cuda               # on a GPU
```

`examples/quickstart.py` builds an Ising square, an XY simple-cubic and a Heisenberg triangular system, checks
the parallel sweep against the sequential reference chain spin by spin, and prints the energy per site; the
expected CPU output is `examples/quickstart.expected.txt`.

## Minimal use

```python
from Spin_Colour_Kernel import system as sy
from Spin_Colour_Kernel.sampler import Sampler

s = sy.heisenberg("fcc", 32)              # model, crystal, side; see system.py for masks, fields, shells
A = Sampler(s, batch=128, device="cuda", seed=1)
A.set_temperature(1.2)                    # one temperature, or one per replica
S = A.init_spins(seed=2)
A.sweep(S, 1000)
print(A.energy(S) / s.n_active)           # energy per site of each replica
```

## Layout

| Path | Content |
|---|---|
| `src/Spin_Colour_Kernel/crystals.py` | Crystals (Bravais lattice + basis) and bond tables from neighbour shells |
| `src/Spin_Colour_Kernel/colouring.py` | Linear colourings, admissibility, periodic-seam condition, clique number |
| `src/Spin_Colour_Kernel/system.py` | System = model + crystal + geometry + couplings; the test matrix |
| `src/Spin_Colour_Kernel/generator.py` | Emits and compiles the Triton kernel of one system |
| `src/Spin_Colour_Kernel/sampler.py` | Sweeps, sequential reference chain, CUDA graphs, observables |
| `src/Spin_Colour_Kernel/bitwise.py` | Bitwise multi-spin-coded Ising mode and its NumPy reference |
| `src/Spin_Colour_Kernel/exp_core.py`, `exp_perf.py`, `width_law*.py`, `chunk_check.py` | The paper's experiments (exactness, colouring control, critical points, throughput, XLA comparison) |
| `src/Spin_Colour_Kernel/vampire_pair.py`, `boris_pair.py`, `cudametro_pair.py` | Paired comparisons with VAMPIRE, BORIS and CUDA-METRO (fetched and built at run time at pinned commits, not vendored) |
| `src/Spin_Colour_Kernel/run.py` | Command-line runner: `python -m Spin_Colour_Kernel.run --exp E1,E3 --out DIR` |
| `tests/` | CPU test suite (Triton interpreter) |
| `examples/` | Quick start with expected output; `library_nanodisk.ipynb`, the pip library on a production nanodisk |
| `src/Spin_Colour_Kernel/api.py`, `src/sck/` | The Python interface: `sck.run`, `check`, `describe`, `write`, `load` |
| `src/Spin_Colour_Kernel/mcp.py`, `agent.py`, `jobs.py`, `runfile.py` | The MCP server for agents: describe, check, write, and launch / follow / fetch runs per backend |
| `src/Spin_Colour_Kernel/simulate.py` | General runs behind the simulation notebook: active-site shapes, Hamiltonian per state, temperature ladder, outputs |
| `ready_for_run/` | The simulation notebook, the run configurator page and the user manuals (English, Spanish; Markdown and PDF) |
| `notebooks/` | `spin_simulation.ipynb` (any supported Hamiltonian, ready to run) and the nanodisk application notebook |
| `remote/` | Scripts that ran the paper's experiments on Modal (H100, A100) and Kaggle (T4), and that fold run outputs into `results-product/` |
| `results-product/Results/` | Tables, figures, significance tests and discussion of the paper's runs |
| `third_party/` | cuIsing 2025 program files (MIT, from the CPC Program Library) used in the specialist comparison |

Comments in the experiment modules cite the revision of the registered experiment protocol that introduced
each test (for example "experiments v04"); the paper describes each experiment in full. The raw run outputs
(about 1.5 GB) are archived separately; `results-product/` holds everything the paper reports.

## Citation

See `CITATION.cff`.

## Licence

MIT (see `LICENSE`). Third-party code keeps its own licence: cuIsing 2025 (MIT) in `third_party/`; VAMPIRE
(GPL-2.0) and BORIS (GPL-3.0) are downloaded and built at run time by the comparison modules and are not
distributed here.
