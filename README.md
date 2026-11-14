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

`ready_for_run/` holds the notebook with a step-by-step user manual in English and Spanish (Markdown and PDF).
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

## Installation

```bash
git clone https://github.com/canamejoy/spin-colour-kernel
cd spin-colour-kernel
pip install -e ".[dev]"
```

The GPU runs of the paper used Python 3.11, `torch==2.5.1` (CUDA 12.4) and `triton==3.1.0`. The CPU test suite
also passes with recent CPU-only torch and triton through the Triton interpreter.

## Test run

```bash
pytest                                                    # 175 tests, CPU only (Triton interpreter), ~25 min
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
| `examples/` | Quick start with expected output |
| `src/Spin_Colour_Kernel/simulate.py` | General runs behind the simulation notebook: active-site shapes, Hamiltonian per state, temperature ladder, outputs |
| `ready_for_run/` | The simulation notebook and its user manuals (English, Spanish; Markdown and PDF) |
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
