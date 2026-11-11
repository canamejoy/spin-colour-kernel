"""Quick start and test run: build three systems, print their linear colourings, check the parallel coloured
sweep against the sequential reference chain spin by spin, and print the energy per site after 20 sweeps.

Runs on a CUDA GPU, or on the CPU through the Triton interpreter:

    TRITON_INTERPRET=1 python examples/quickstart.py --device cpu     # about a minute, no GPU needed
    python examples/quickstart.py --device cuda

The expected output of the CPU run (fixed seeds) is in examples/quickstart.expected.txt. On a GPU the
"parallel == sequential" column must read True for every system; the energies can differ in the last digits
from the CPU run, because float32 exp is not bit-identical across devices.
"""

import argparse

import torch

from Spin_Colour_Kernel import system as sy
from Spin_Colour_Kernel.sampler import Sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    systems = [sy.ising("square", 8), sy.xy("sc", 4), sy.heisenberg("triangular", 6)]
    for s in systems:
        A = Sampler(s, 2, device=a.device, seed=11)
        R = Sampler(s, 2, device=a.device, seed=11)
        for X in (A, R):
            X.set_temperature(1.5)
        S = A.init_spins(seed=2)
        S_ref = S.clone()
        A.sweep(S, 3)
        R.reference_sweep(S_ref, 3)
        same = bool(torch.equal(S, S_ref))
        A.sweep(S, 17)
        e = A.energy(S).double().cpu() / s.n_active
        print(f"{s.name:24s} colours {s.colouring.n_colours}  parallel == sequential after 3 sweeps: {same}  "
              f"energy/site after 20 sweeps: {e[0]:+.6f} {e[1]:+.6f}")


if __name__ == "__main__":
    main()
