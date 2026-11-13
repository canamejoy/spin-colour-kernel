"""Cooling ladder of the magnetic nanodisk on the generated kernel.

One state is one Hamiltonian (seven parameters), one compiled kernel and one batch of replicas. Every
replica follows the same ladder from T_start to T_end (kelvin); at each temperature it thermalizes
n_therm sweeps and measures every `every` sweeps during n_meas sweeps. The accumulators hold
fluctuations about a reference taken once per temperature, so float cancellation cannot destroy
C_v and chi, and `reduce_obs` gives the metrics from

    obs[t] = (<dE>, <dE^2>, <dM>, <dM^2>, <Mz>, <Q>, E_ref, M_ref)      shape (n_T, 8, B)

with E the total energy (units of Jex), M = |sum_i s_i| / N_active, Mz its z component, and Q the
Berg-Luscher charge of the middle layer. Design choices:
- the update is the generated multi-colour kernel (any of the sc shells 1-4, colouring searched and
  verified), launched from CUDA graphs of `every` sweeps;
- observables are read every `every` sweeps, not after every sweep (they cost more than a sweep);
- energies and magnetizations are accumulated in float64.

Run as a module to process several states on one GPU and write one .npz per state:
    python -m Spin_Colour_Kernel.anneal --config run.json --states 0,2 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import system as sy
from .sampler import Sampler

KB = 0.086173404     # meV / K (Jex = 1 sets the energy unit)
PARAM_NAMES = ("Jex2", "Jex3", "Jex4", "Kan1", "KanS", "Hex", "KDM")


def build_state(params, Rd: float, L: int, angle_deg: float, shells=(1,)):
    """The nanodisk System for one parameter row [Jex2, Jex3, Jex4, Kan1, KanS, Hex, KDM]."""
    J2, J3, J4, K1, KS, H, D = (float(x) for x in params)
    return sy.nanodot(Rd=Rd, L=L, Jex=1.0, Kan1=K1, Kan2=0.0, KanS=KS, Hex=H, KDM=D, gamma_deg=angle_deg,
                      shells=tuple(shells), J_shells={2: J2, 3: J3, 4: J4})


def active_shells(rows) -> list[int]:
    """Union over states of the shells with a non-zero coupling: one colouring serves every state."""
    out = {1}
    for r in rows:
        for k, v in zip((2, 3, 4), r[:3]):
            if float(v) != 0.0:
                out.add(k)
    return sorted(out)


def check_exact(s, device, sweeps: int = 3, T: float = 0.5, seed: int = 1) -> int:
    """Differing spins between the coloured kernel and the sequential reference chain (must be 0)."""
    A, R = Sampler(s, 2, device, seed), Sampler(s, 2, device, seed)
    A.set_temperature(T)
    R.set_temperature(T)
    S1 = A.init_spins(seed=seed)
    S2 = S1.clone()
    A.sweep(S1, sweeps)
    R.reference_sweep(S2, sweeps)
    return int((S1 != S2).sum())


def _topo(layer: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Berg-Luscher charge of one layer; layer (B, 3, N, N) indexed [y, x], valid (N-1, N-1)."""
    s = layer.permute(0, 2, 3, 1)
    s00, s10, s01, s11 = s[:, :-1, :-1], s[:, :-1, 1:], s[:, 1:, :-1], s[:, 1:, 1:]

    def om(a, b, c):
        num = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
        return 2.0 * torch.atan2(num, den)
    w = om(s00, s10, s11) + om(s00, s11, s01)
    return (w * valid).sum(dim=(1, 2)) / (4.0 * math.pi)


class Observer:
    """E (float64), mean magnetization vector and middle-layer Q per replica, safe inside a CUDA graph.

    Same quantities as Sampler.energy / Sampler.magnetization, but every constant (bond matrices, open-
    boundary masks, field, anisotropy) is built once here: inside a graph capture no host-to-device copy
    may run, and those methods build their constants on every call.
    """

    def __init__(self, A: Sampler):
        s = A.sys
        dev = A.k1.device
        self.s, self.n_act = s, float(s.n_active)
        self.bonds = []
        dims = {0: 5, 1: 4, 2: 3}
        for i, bd in enumerate(s.crystal.bonds):
            M = s.bond_matrix(i)
            if not M.any():
                continue
            valid = torch.ones(s.shape[1:], dtype=torch.float64, device=dev)
            shifts = []
            for k in range(s.crystal.D):
                d = bd.delta[k]
                if d == 0:
                    continue
                shifts.append((-d, dims[k] - 1))
                if not s.periodic[k]:
                    idx = torch.arange(s.box[k], device=dev)
                    ok = ((idx + d) >= 0) & ((idx + d) < s.box[k])
                    shp = [1, 1, 1]
                    shp[2 - k] = s.box[k]
                    valid = valid * ok.to(torch.float64).reshape(shp)
            self.bonds.append((bd.alpha, bd.alpha2, shifts, valid,
                               torch.as_tensor(M, dtype=torch.float64, device=dev)))
        self.h = (None if s.h is None else
                  torch.as_tensor(s.h, dtype=torch.float64, device=dev).reshape(1, -1, 1, 1, 1, 1))
        self.k1 = None if s.cubic_k1 is None else A.k1.double()
        self.k2 = float(s.cubic_k2)
        self.mid = s.L3 // 2
        disk = torch.as_tensor(s.site_mask()[0, self.mid], device=dev)
        self.valid2d = (disk[:-1, :-1] & disk[:-1, 1:] & disk[1:, :-1] & disk[1:, 1:]).double()

    def __call__(self, S: torch.Tensor):
        S64 = S.double()
        v = torch.zeros_like(S64)
        for a, a2, shifts, valid, M in self.bonds:
            src = S64[:, :, a2]
            for sh, dim in shifts:
                src = torch.roll(src, shifts=sh, dims=dim)
            v[:, :, a] += torch.einsum("kl,blzyx->bkzyx", M, src * valid)
        E = -0.5 * (S64 * v).sum(dim=(1, 2, 3, 4, 5))
        if self.h is not None:
            E = E - (S64 * self.h).sum(dim=(1, 2, 3, 4, 5))
        if self.k1 is not None:
            x, y, z = S64[:, 0], S64[:, 1], S64[:, 2]
            E = E + (self.k1 * ((x * y) ** 2 + (x * z) ** 2 + (y * z) ** 2)
                     + self.k2 * (x * y * z) ** 2).sum(dim=(1, 2, 3, 4))
        Mv = S64.sum(dim=(2, 3, 4, 5)) / self.n_act
        Q = _topo(S64[:, :, 0, self.mid], self.valid2d)
        return E, Mv, Q


def anneal_state(s, replicas: int, temps_K, n_therm: int, n_meas: int, every: int, seed: int,
                 device="cuda", log=print):
    """Run the ladder for one state; returns obs (n_T, 8, B), mvec (n_T, B, 3), the final middle-layer
    s_z of replica 0 per temperature (n_T, N, N), and timings."""
    A = Sampler(s, replicas, device, seed)
    S = A.init_spins(seed=seed)
    dev = S.device
    n_act = float(s.n_active)
    observe = Observer(A)
    k = max(1, int(every))
    cuda = str(dev).startswith("cuda")
    big = 10 * k
    nT = len(temps_K)
    obs = np.zeros((nT, 8, replicas))
    mvec = np.zeros((nT, replicas, 3), np.float32)
    snaps = np.zeros((nT,) + tuple(observe.valid2d.shape[i] + 1 for i in range(2)), np.float32)
    f64 = dict(dtype=torch.float64, device=dev)
    E_ref, M_ref = torch.zeros(replicas, **f64), torch.zeros(replicas, **f64)
    acc = torch.zeros(6, replicas, **f64)                     # dE, dE^2, dM, dM^2, Mz, Q
    accv = torch.zeros(replicas, 3, **f64)

    def chunk():           # k sweeps, then one observation folded into the accumulators
        A.sweep(S, k)
        E, Mv, Q = observe(S)
        dE = E - E_ref
        dM = Mv.norm(dim=1) - M_ref
        acc.add_(torch.stack([dE, dE * dE, dM, dM * dM, Mv[:, 2], Q]))
        accv.add_(Mv)

    A.set_temperature(KB * float(temps_K[0]))
    if cuda:
        # the thermalization block and the measure-and-accumulate step each become one CUDA graph: no
        # host launches, no host-device copies inside the ladder (beta is written in place)
        rep_big = A.graph(S, big)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            chunk()                                            # warm-up outside the capture
        torch.cuda.current_stream(dev).wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            chunk()
        rep_obs = g.replay
    else:
        rep_big, rep_obs = (lambda: A.sweep(S, big)), chunk

    t0 = time.perf_counter()
    for t, TK in enumerate(temps_K):
        A.set_temperature(KB * float(TK))
        for _ in range(n_therm // big):
            rep_big()
        if n_therm % big:
            A.sweep(S, n_therm % big)
        E, Mv, _ = observe(S)
        E_ref.copy_(E)
        M_ref.copy_(Mv.norm(dim=1))
        acc.zero_()
        accv.zero_()
        m = max(1, n_meas // k)
        for _ in range(m):
            rep_obs()
        a = (acc / m).cpu().numpy()
        obs[t, :6] = a
        obs[t, 6] = E_ref.cpu().numpy()
        obs[t, 7] = M_ref.cpu().numpy()
        mvec[t] = (accv / m).float().cpu().numpy()
        snaps[t] = S[0, 2, 0, observe.mid].float().cpu().numpy()
        if t % max(1, nT // 10) == 0 or t == nT - 1:
            log(f"    T = {TK:7.3f} K   <E>/site = {(obs[t, 6].mean() + obs[t, 0].mean()) / n_act:+.4f}"
                f"   |M| = {obs[t, 7].mean() + obs[t, 2].mean():.4f}   {time.perf_counter() - t0:7.1f} s")
    if cuda:
        torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    sweeps = nT * (n_therm + m * k)
    return dict(obs=obs, mvec=mvec, snaps=snaps, seconds=dt, sweeps_per_chain=sweeps,
                flips_per_ns=replicas * n_act * sweeps / dt / 1e9)


def reduce_obs(obs, temps, n_spins, kB: float = KB):
    """Reduction: accumulators (n_T, 8, B) -> per-temperature scalars."""
    obs = np.asarray(obs)
    assert obs.ndim == 3 and obs.shape[1] == 8, f"expected (n_T, 8, batch), got {obs.shape}"
    dE, dE2, dM, dM2, Mz, Q, E_ref, M_ref = [obs[:, i] for i in range(8)]
    T = np.asarray(temps)[:, None]
    var_E = dE2 - dE ** 2
    var_M = dM2 - dM ** 2
    E = dE + E_ref
    M = dM + M_ref
    Cv = var_E / (kB * T ** 2 * n_spins)
    chi = var_M / (kB * T)
    return dict(T=np.asarray(temps), E=E.mean(1) / n_spins, E_sd=E.std(1) / n_spins, M=M.mean(1),
                M_sd=M.std(1), Mz=Mz.mean(1), Cv=Cv.mean(1), chi=chi.mean(1), Q=Q.mean(1), Q_sd=Q.std(1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--states", required=True, help="comma list of state indices")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    rows = cfg["params"]
    shells = active_shells(rows)
    temps = np.linspace(cfg["T_start"], cfg["T_end"], int(cfg["n_temps"]))
    n_therm = int(cfg["sweeps"]) * 2 // 3
    n_meas = int(cfg["sweeps"]) - n_therm
    for i in (int(x) for x in a.states.split(",") if x):
        name = cfg["names"][i]
        s = build_state(rows[i], cfg["Rd"], cfg["layers"], cfg["angle"], shells)
        d = check_exact(s, a.device)
        print(f"[{name}] {s.colouring.n_colours} colours ({s.colouring.describe()}), "
              f"exactness vs sequential chain: {d} differing spins", flush=True)
        if d:
            raise SystemExit(f"[{name}] the coloured kernel does not reproduce the sequential chain")
        r = anneal_state(s, int(cfg["replicas"]), temps, n_therm, n_meas, int(cfg["every"]),
                         int(cfg["seed"]) + i, a.device, log=lambda m: print(f"[{name}]" + m, flush=True))
        np.savez_compressed(a.out / f"state_{name}.npz", params=np.asarray(rows[i], np.float32),
                            obs=r["obs"], M=np.transpose(r["mvec"], (1, 0, 2)), snaps=r["snaps"],
                            temperatures=temps.astype(np.float32), n_active=s.n_active,
                            seconds=r["seconds"], flips_per_ns=r["flips_per_ns"],
                            colours=s.colouring.n_colours)
        print(f"[{name}] done in {r['seconds']:.1f} s, {r['flips_per_ns']:.2f} spin flips/ns", flush=True)


if __name__ == "__main__":
    main()
