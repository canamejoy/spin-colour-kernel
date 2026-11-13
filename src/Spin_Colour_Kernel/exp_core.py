"""Experiments E1-E3 of experiments v03 (exactness, colouring control, colourings vs clique)."""

from __future__ import annotations

import itertools
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import colouring as co
from . import crystals as cr
from . import system as sy
from .sampler import Sampler


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import jax
        env["jax"] = jax.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        env["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        try:
            env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip().splitlines()[0]
        except Exception:
            pass
    return env


def save(out: Path, name: str, payload: dict):
    out.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": name, "environment": environment(), **payload}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


# ------------------------------------------------------------------ E1
def e1_bitwise(out: Path, sweeps: int = 50, seeds=(1001, 1002, 1003), device="cuda", groups=(1, 4, 8),
               tag: str = "E1bit", independent: bool = False):
    """Experiments v06, E1 bitwise mode: the multi-spin Ising kernel on the GPU against its sequential reference
    (`bitwise.reference_sweep`: per replica, same colour order, same uniform per word/site/sweep, same table),
    B = 64 (two words at two temperatures), Metropolis and Barker, for every words_per_program in `groups`."""
    from . import bitwise as bw
    rng = np.random.default_rng(0)
    mask_sq = rng.random((1, 1, 8, 8)) > 0.25
    mask_sc = np.ones((1, 3, 3, 4), bool)
    mask_sc[0, 1, 1, 1:3] = False

    def systems():
        return [sy.ising("square", 8), sy.ising("triangular", 6), sy.ising("sc", 4), sy.ising("fcc", 4),
                sy.System("Ising square masked h", 1, cr.make("square"), (8, 8), (True, True), {1: 1.0}, h=(0.3,),
                          mask=mask_sq),
                sy.System("Ising sc open masked", 1, cr.make("sc"), (4, 3, 3), (False,) * 3, {1: 1.0}, mask=mask_sc)]
    rows = []
    for acceptance in ("metropolis", "barker"):
        for s in systems():
            s.acceptance = acceptance
            Tc = s.T_ref or 3.0
            for G in groups:
                for seed in seeds:
                    A = Sampler(s, 64, device, seed, bitwise=True, words_per_program=G, independent=independent)
                    A.set_temperature(torch.tensor([2.0] * 32 + [4.0] * 32, device=A.device) * (Tc / 3.0))
                    S = A.init_spins(seed=seed)
                    ref = A.unpack(S).cpu().numpy()[:, 0].copy()
                    first = None
                    for k in range(sweeps):
                        sweep = int(A.state[1])
                        A.sweep(S)
                        ref = bw.reference_sweep(s, ref, A.ptab.cpu().numpy(), int(A.state[0]), sweep,
                                                 **({"independent": True} if independent else {}))
                        d = int((A.unpack(S).cpu().numpy()[:, 0] != ref).sum())
                        if d and first is None:
                            first = (k + 1, d)
                            break
                    rows.append({"system": s.name, "box": list(s.box), "acceptance": acceptance, "words_per_program": G,
                                 "seed": seed, "sweeps": sweeps, "first_difference": first})
            bad = [r for r in rows if r["system"] == s.name and r["acceptance"] == acceptance and r["first_difference"]]
            print(f"E1bit {s.name:24s} {acceptance:10s} runs {len(groups) * len(seeds)} differing {len(bad)}", flush=True)
    n_bad = sum(1 for r in rows if r["first_difference"])
    return save(out, tag, {"rows": rows, "runs": len(rows), "runs_with_difference": n_bad, "independent": independent,
                           "verdict": "supported" if n_bad == 0 else "not supported"})


def e1(out: Path, sweeps: int = 1000, seeds=(1, 2, 3), batch: int = 4, device="cuda"):
    """Spin-by-spin exactness of the parallel chain against the sequential reference (H1)."""
    rows = []
    for s in sy.matrix():
        Tref = s.T_ref or 1.0
        temps = [0.5 * Tref, Tref, 2.0 * Tref]
        diffs, first, done = 0, None, 0
        t0 = time.time()
        for seed in seeds:
            for T in temps:
                A, R = Sampler(s, batch, device, seed), Sampler(s, batch, device, seed)
                A.set_temperature(T)
                R.set_temperature(T)
                S1 = A.init_spins(seed=seed)
                S2 = S1.clone()
                for k in range(sweeps):
                    A.sweep(S1)
                    R.reference_sweep(S2)
                    d = int((S1 != S2).sum())
                    if d:
                        diffs += d
                        if first is None:
                            idx = (S1 != S2).nonzero()[0].tolist()
                            first = {"seed": seed, "T": T, "sweep": k + 1, "index": idx}
                        break
                    done += 1
        rows.append({"system": s.name, "n": s.n, "crystal": s.crystal.name, "box": list(s.box),
                     "colours": s.colouring.n_colours, "colouring": s.colouring.describe(),
                     "active_sites": s.n_active, "seeds": len(seeds), "temperatures": temps,
                     "sweeps_compared": done, "differing_spins": diffs, "first_failure": first,
                     "seconds": time.time() - t0})
        print(f"E1 {s.name:34s} diffs={diffs} sweeps={done} {time.time() - t0:.0f}s", flush=True)
    ok = all(r["differing_spins"] == 0 and r["sweeps_compared"] == len(seeds) * 3 * sweeps for r in rows)
    return save(out, "E1", {"rows": rows, "verdict": "supported" if ok else "not supported",
                            "params": {"sweeps": sweeps, "seeds": list(seeds), "batch": batch}})


# ------------------------------------------------------------------ E2
def exact_energy_distribution(s: sy.System, T: float) -> dict:
    """Exact Boltzmann distribution of the energy of a small periodic Ising system, by enumeration."""
    N = s.n_active
    bonds = []
    L1, L2 = s.L1, s.L2
    for x in range(L1):
        for y in range(L2):
            i = y * L1 + x
            for bd in s.crystal.bonds:
                j = ((y + bd.delta[1]) % L2) * L1 + (x + bd.delta[0]) % L1
                bonds.append((i, j, s.J[bd.shell]))
    states = np.array(list(itertools.product((-1, 1), repeat=N)), dtype=np.int64)
    E = np.zeros(len(states))
    for i, j, J in bonds:
        E -= 0.5 * J * states[:, i] * states[:, j]
    levels, counts = np.unique(np.round(E, 6), return_counts=True)
    w = counts * np.exp(-(levels - levels.min()) / T)
    return {"levels": levels, "p": w / w.sum()}


def integrated_tau(x: np.ndarray, c: float = 6.0) -> float:
    """Integrated autocorrelation time with Sokal's automatic window; x has shape (time, chains)."""
    x = x - x.mean(axis=0)
    n = x.shape[0]
    if not np.all(np.isfinite(x)) or float((x * x).sum()) == 0.0:
        return 1.0      # a constant series (frozen or period-two dynamics) carries no autocorrelation time
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
    acf /= acf[0]
    tau = 1.0
    if not np.isfinite(acf).all():
        return 1.0
    for M in range(1, n):
        tau = 1.0 + 2.0 * acf[1:M + 1].sum()
        if M >= c * tau:
            break
    return max(float(tau), 0.5)


def holm(p: list[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)
    adj = np.zeros(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def e2(out: Path, seeds=tuple(range(1, 11)), samples: int = 100_000, batch: int = 1000, device="cuda",
       alpha: float = 0.01, acceptance: str = "metropolis", tag: str = "E2"):
    """Coloured kernel and all-sites (synchronous) update against exact enumeration (H1b).

    `acceptance="barker"` is the diagnostic E2b: Metropolis with a deterministic scan accepts every
    dE = 0 flip with probability one, which traps chains in deterministic cycles on lattices this small
    (found in E2, 2026-11-06); the heat-bath rule draws a coin there and removes the trap.
    """
    from scipy import stats
    cases = [(sy.ising("square", 4), 2.0 / math.log(1 + math.sqrt(2))),
             (sy.ising("triangular", 3), 4.0 / math.log(3.0))]
    for s, _ in cases:
        s.acceptance = acceptance
    rows, hist = [], {}
    for s, T in cases:
        ex = exact_energy_distribution(s, T)
        lv = ex["levels"]
        for rule in ("coloured", "all sites"):
            stats_rows = []
            pooled = np.zeros(len(lv))
            for seed in seeds:
                A = Sampler(s, batch, device, seed)
                A.set_temperature(T)
                S = A.init_spins(seed=seed)
                step = A.sweep if rule == "coloured" else A.sync_sweep
                step(S, 500)
                pilot = []
                for _ in range(400):
                    step(S, 1)
                    pilot.append(A.energy(S).cpu().numpy())
                tau = integrated_tau(np.array(pilot))
                thin = max(1, int(math.ceil(2 * tau)))
                per = int(math.ceil(samples / batch))
                Es = []
                for _ in range(per):
                    step(S, thin)
                    Es.append(A.energy(S).cpu().numpy())
                Es = np.concatenate(Es)[:samples]
                idx = np.searchsorted(lv, np.round(Es, 6))
                idx = np.clip(idx, 0, len(lv) - 1)
                bad = int((np.abs(lv[idx] - np.round(Es, 6)) > 1e-4).sum())
                obs = np.bincount(idx, minlength=len(lv)).astype(float)
                pooled += obs
                exp_ = ex["p"] * obs.sum()
                # merge bins with expected count below five into their neighbour
                o2, e2_ = [], []
                co_, ce = 0.0, 0.0
                for o, e in zip(obs, exp_):
                    co_ += o
                    ce += e
                    if ce >= 5:
                        o2.append(co_)
                        e2_.append(ce)
                        co_, ce = 0.0, 0.0
                if ce > 0 and e2_:
                    o2[-1] += co_
                    e2_[-1] += ce
                k = len(o2)
                chi2, p = stats.chisquare(o2, e2_) if k > 1 else (0.0, 1.0)
                # Cohen's w; Cramer's V = sqrt(chi2 / (n (k - 1))) is bounded by 1 only for contingency tables
                w = math.sqrt(chi2 / obs.sum())
                stats_rows.append({"seed": seed, "tau": tau, "thin": thin, "chi2": float(chi2), "p": float(p),
                                   "bins": k, "cohens_w": w, "off_level_samples": bad})
            padj = holm([r["p"] for r in stats_rows])
            for r, pa in zip(stats_rows, padj):
                r["p_holm"] = pa
            rejected = sum(pa < alpha for pa in padj)
            hist[f"{s.name}|{rule}"] = {"levels": lv.tolist(), "exact": ex["p"].tolist(),
                                        "sampled": (pooled / pooled.sum()).tolist()}
            rows.append({"lattice": s.name, "rule": rule, "T": T, "seeds": stats_rows, "rejected_seeds": rejected,
                         "median_chi2": float(np.median([r["chi2"] for r in stats_rows])),
                         "median_cohens_w": float(np.median([r["cohens_w"] for r in stats_rows]))})
            print(f"E2 {s.name:20s} {rule:9s} rejected {rejected}/{len(seeds)}", flush=True)
    col = [r for r in rows if r["rule"] == "coloured"]
    ctl = [r for r in rows if r["rule"] != "coloured"]
    ok = all(r["rejected_seeds"] == 0 for r in col) and all(r["rejected_seeds"] == len(seeds) for r in ctl)
    return save(out, tag, {"rows": rows, "histograms": hist, "verdict": "supported" if ok else "see rows",
                           "params": {"seeds": list(seeds), "samples": samples, "batch": batch, "alpha": alpha,
                                      "acceptance": acceptance}})


# ------------------------------------------------------------------ E3
def e3(out: Path):
    """Minimum linear colouring, clique number, independent admissibility check (H2)."""
    rows = []
    for name, sh in [("square", (1,)), ("sc", (1,)), ("sc", (1, 2)), ("sc", (1, 2, 3, 4)), ("bcc", (1,)),
                     ("fcc", (1,)), ("triangular", (1,)), ("honeycomb", (1,)), ("kagome", (1,))]:
        c = cr.make(name, sh)
        t0 = time.time()
        col = co.search(c)
        w = co.clique_number(c)
        # independent check on the smallest periodic box that meets the seam condition (8)
        L = next(L for L in range(2, 40) if co.seam_ok(col, (L,) * c.D, (True,) * c.D))
        box = (max(L, 2 * L if L < 3 else L),) * c.D
        ok = co.admissible(col, c) and co.verify_on_box(col, c, box, (True,) * c.D)
        rows.append({"crystal": name, "shells": list(sh), "bonds": len(c.bonds), "group": list(col.m),
                     "colouring": col.describe(), "linear_colours": col.n_colours, "clique_number": w,
                     "admissible": ok, "optimality_certified": col.n_colours == w, "check_box": list(box),
                     "seconds": time.time() - t0})
        print(f"E3 {name:10s} {sh} colours={col.n_colours} clique={w} ok={ok}", flush=True)
    return save(out, "E3", {"rows": rows, "verdict": "supported" if all(r["admissible"] for r in rows) else
                            "not supported"})
