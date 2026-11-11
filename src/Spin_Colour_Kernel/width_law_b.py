"""Experiment E5b: the thread-count rule of XLA's loop emitter, read where it applies.

E5 measured the widest vector load on grids of at most 19,200 elements and saw only scalar loads.
XLA's own source explains why (openxla/xla main, read 2026-11-06):

    xla/service/gpu/gpu_fusible.cc, ComputeLoopFusionConfig:
        if (num_elements >= n_threads_max && !MayCausePerformanceDropIfUnrolled(fusion))
            unroll_factor = std::gcd(num_elements, MaxUnrollFactor(&analysis));   // 4 before Blackwell
        n_threads_max = threads_per_core_limit * core_count                         // T4: 1024 * 40
    xla/service/gpu/launch_dimensions.cc, CalculateLaunchDimensions:
        num_elements = CeilOfRatio(num_elements, unroll_factor);   // one thread per unroll elements

So each loop fusion over N elements runs u = gcd(N, 4) elements per thread, and N / u threads, but
only when N reaches the device's resident-thread count; below it u = 1. Equation (11) is this rule
without its threshold, and every E5 grid sat below the threshold.

E5b reads three things per grid:
  1. the unroll factors XLA logs (`VLOG(2) "Unroll factor: k"`, TF_CPP_VMODULE=gpu_fusible=2);
  2. every loop fusion of the optimized HLO with its first output shape, and the factor the rule
     predicts for it: gcd(N_f, 4) if N_f >= n_threads_max else 1;
  3. the time per spin update of the XLA baseline (the same jnp.roll step as E5) and of the generated
     Triton kernel on the same model, crystal and box.

Run as its own process (logging and dump flags must be set before JAX starts):
    python -m Spin_Colour_Kernel.width_law_b --out results/ [--quick]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

UNROLL = re.compile(rb"Unroll factor: (\d+)")
MARK = "E5B-GRID-MARK"
#  %fusion.3 = f32[5,3,1,15,15,15]{5,4,3,2,1,0} fusion(...), kind=kLoop, calls=...
#  %fusion.4 = (f32[...]{...}, pred[...]{...}) fusion(...), kind=kLoop, ...
FUSION = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*\(?\s*(\w+)\[([\d,]*)\][^\n]*?\bfusion\([^\n]*\bkind=(k\w+)",
                    re.M)
BITS = {"pred": 8, "s8": 8, "u8": 8, "f16": 16, "bf16": 16, "s16": 16, "u16": 16, "f32": 32, "s32": 32,
        "u32": 32, "f64": 64, "s64": 64, "u64": 64}


def rule(n_elements: int, threshold: int, bits: int = 32) -> int:
    u = math.gcd(n_elements, 4) if n_elements >= threshold else 1
    return max(u, math.ceil(8 / bits))


def h4_cost(rows, factor="layer-heis", boots: int = 2000, seed: int = 0):
    """H4(b): paired cost of a u = 4 grid against its odd (u = 1) neighbours in L1, per timing round.

    rows are E5c rows of one sweep over L1 (odd extents elsewhere, so gcd(N, 4) = gcd(L1, 4)). Per round
    r the statistic is log t(L1) - mean(log t(L1 - 1), log t(L1 + 1)) over the neighbours present; the
    rounds are interleaved over grids, so a device drift is common to both sides. Returns one entry per
    u = 4 grid plus the pooled mean, each with a percentile bootstrap interval over rounds.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    by = {r["box"][0]: np.log(np.asarray(r["xla_ns_per_update"], float)) for r in rows if r["factor"] == factor}
    out, stats = [], []
    for L1 in sorted(by):
        if L1 % 4:
            continue
        nb = [by[x] for x in (L1 - 1, L1 + 1) if x in by]
        if not nb:
            continue
        d = by[L1] - np.mean(nb, axis=0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boots)]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out.append({"L1": L1, "log_ratio": float(d.mean()), "low": float(lo), "high": float(hi),
                    "above_zero": bool(lo > 0)})
        stats.append(d)
    if stats:
        P = np.mean(stats, axis=0)
        bs = [P[rng.integers(0, len(P), len(P))].mean() for _ in range(boots)]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        pooled = {"log_ratio": float(P.mean()), "low": float(lo), "high": float(hi)}
    else:
        pooled = None
    return {"grids": out, "pooled": pooled,
            "supported": bool(out) and all(g["above_zero"] for g in out)}


def grids_c(quick: bool):
    """E5c: N of 10^6 and more, where a sweep is bound by memory, not by launch latency. Odd extents
    everywhere except the factor varied, so gcd(N, 4) is set by that factor alone."""
    if quick:
        return [("layer", 63, 1, 1, (L1, 31, 31)) for L1 in (63, 64)]
    g = [("layer", 63, 1, 1, (L1, 31, 31)) for L1 in range(61, 73)]
    g += [("layer-heis", 21, 3, 1, (L1, 31, 31)) for L1 in range(61, 73)]
    g += [("B", B, 1, 1, (31, 31, 31)) for B in range(33, 49)]
    g += [("b", 21, 1, b, (31, 31, 31)) for b in (1, 2, 4)]
    return g


def make_baseline_step(n, b, box, gid):
    """The E5 baseline: one Metropolis sweep in jnp.roll form, jitted (also profiled by e5_profile)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Spin_Colour_Kernel import colouring as co
    from Spin_Colour_Kernel import crystals as cr
    crystal_for_b = {1: "sc", 2: "bcc", 4: "fcc"}
    c = cr.make(crystal_for_b[b])
    col = co.search(c)
    L1, L2, L3 = box
    cm = np.zeros((b, L3, L2, L1), np.int32)
    for al in range(b):
        for z, y, x in itertools.product(range(L3), range(L2), range(L1)):
            cm[al, z, y, x] = col.colour((x, y, z), al)
    cmj = jnp.asarray(cm)
    nc = col.n_colours

    def step(S, key, beta):
        for colour in range(nc):
            vs = []
            for al in range(b):
                v = jnp.zeros_like(S[:, :, 0])
                for bd in c.bonds:
                    if bd.alpha != al:
                        continue
                    src = S[:, :, bd.alpha2]
                    for k, ax in zip(range(3), (4, 3, 2)):
                        if bd.delta[k]:
                            src = jnp.roll(src, -bd.delta[k], axis=ax)
                    v = v + src
                vs.append(v)
            V = jnp.stack(vs, axis=2)
            key, k1, k2 = jax.random.split(key, 3)
            if n == 1:
                T = -S
            else:
                g = jax.random.normal(k1, S.shape, S.dtype)
                T = g / jnp.linalg.norm(g, axis=1, keepdims=True)
            dE = -jnp.sum((T - S) * V, axis=1)
            u = jax.random.uniform(k2, dE.shape, dE.dtype)
            acc = (u < jnp.exp(-beta * dE)) & (cmj[None] == colour)
            S = jnp.where(acc[:, None], T, S)
        return S, key

    step.__name__ = f"wlb{gid:04d}"
    return jax.jit(step)


CONF_NB = ((1, 1), (2, 1), (3, 1), (1, 2), (1, 4), (3, 4))
CONF_L1 = (75, 76, 77, 79, 80, 81, 83, 84, 85)     # u = 4 at 76, 80, 84; odd neighbours L1 +- 1
MARGIN_H4 = math.log(1.3)


def grids_conf(threshold: int, quick: bool = False):
    """Experiments v04, E5 confirmatory grids (none used in E5, E5b or E5c).

    (a) for each (n, b) of CONF_NB, B = 4 and an odd cube L^3, so the element count 4 n b L^3 is a multiple
    of 4: the largest L with N <= 0.8 tC ("below") and the smallest with N >= 1.25 tC ("above").
    (b) the Heisenberg sweep B = 21, L1 x 31^2 over CONF_L1 ("layer-heis") and the Ising one, B = 63
    ("layer"), reported only."""
    g = []
    for n, b in (CONF_NB[:2] if quick else CONF_NB):
        per = 4 * n * b
        lo = max(L for L in range(3, 401, 2) if per * L ** 3 <= 0.8 * threshold)
        hi = min(L for L in range(3, 401, 2) if per * L ** 3 >= 1.25 * threshold)
        g.append(("below", 4, n, b, (lo, lo, lo)))
        g.append(("above", 4, n, b, (hi, hi, hi)))
    L1s = (75, 76, 77) if quick else CONF_L1
    g += [("layer-heis", 21, 3, 1, (L1, 31, 31)) for L1 in L1s]
    g += [("layer", 63, 1, 1, (L1, 31, 31)) for L1 in L1s]
    return g


GEN_L1 = tuple(range(99, 110))      # v15: held-out L1 (u = 4 at 100, 104, 108), none used before


def grids_gen(threshold: int, quick: bool = False):
    """Experiments v15, H4 reframed: the generated kernel against the XLA baseline on held-out grids.

    The (n, b) cubes just below and above the device threshold (as in v04) and the Heisenberg (B = 21) and
    Ising (B = 63) sweeps over L1 x 31^2 for L1 in GEN_L1."""
    g = [x for x in grids_conf(threshold, quick) if x[0] in ("below", "above")]
    L1s = GEN_L1[:3] if quick else GEN_L1
    g += [("layer-heis", 21, 3, 1, (L1, 31, 31)) for L1 in L1s]
    g += [("layer", 63, 1, 1, (L1, 31, 31)) for L1 in L1s]
    return g


GENB_B = tuple(range(1, 17)) + (20, 24, 28, 31, 32, 33, 40, 48, 56, 63, 64)
GENB_BOX = (27, 27, 27)     # odd composite side: 3 colours for sc, while gcd(N, 4) = gcd(B n, 4) follows B
JUMP = math.log(1.05)


def grids_genb(quick: bool = False):
    """Experiments v16: the same geometry (sc 27^3) for B = 1..64, Heisenberg (n = 3) and Ising (n = 1)."""
    Bs = (1, 2, 3, 4) if quick else GENB_B
    return [(f"B-{'heis' if n == 3 else 'ising'}", B, n, 1, GENB_BOX) for n in (3, 1) for B in Bs]


def genb_verdict(rows, boots: int = 2000, seed: int = 0):
    """v16 criterion per device: (i) the generated kernel's median cost per update never rises by more than 5%
    from one B to the next (upward jumps of log cost above log 1.05), counted beside the same count for XLA;
    (ii) the generated kernel is faster at every B (lower bound of the paired log speed-up above zero)."""
    import numpy as np
    sp = gen_verdict(rows, boots, seed)
    out = {}
    for fac in sorted({r["factor"] for r in rows}):
        rs = sorted((r for r in rows if r["factor"] == fac and r.get("generated_ns_per_update")), key=lambda r: r["B"])
        xm = np.log([float(np.median(r["xla_ns_per_update"])) for r in rs])
        gm = np.log([float(np.median(r["generated_ns_per_update"])) for r in rs])
        dx, dg = np.diff(xm), np.diff(gm)
        out[fac] = {"B": [r["B"] for r in rs], "xla_median_ns": np.exp(xm).tolist(), "generated_median_ns": np.exp(gm).tolist(),
                    "xla_upward_jumps": int((dx > JUMP).sum()), "generated_upward_jumps": int((dg > JUMP).sum()),
                    "xla_max_jump": float(dx.max()) if len(dx) else None,
                    "generated_max_jump": float(dg.max()) if len(dg) else None}
    return {"per_model": out, "speedup": sp,
            "generated_smooth": all(v["generated_upward_jumps"] == 0 for v in out.values()),
            "generated_faster_everywhere": sp["all_faster"]}


def gen_verdict(rows, boots: int = 2000, seed: int = 0):
    """v15 criterion: per grid the paired log speed-up log t_xla - log t_gen over interleaved rounds, with a
    percentile bootstrap interval; the generated kernel is faster on a grid when the lower bound is above
    zero. Also the spread of each side's median cost over the L1 sweeps (max / min), which measures how much
    the cost depends on the grid shape."""
    import numpy as np
    rng = np.random.default_rng(seed)
    per, allp = [], []
    for r in rows:
        x = np.log(np.asarray(r["xla_ns_per_update"], float))
        g = np.log(np.asarray(r["generated_ns_per_update"], float))
        if not len(g) or len(g) != len(x):
            continue
        d = x - g
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boots)]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        per.append({"factor": r["factor"], "B": r["B"], "n": r["n"], "b": r["b"], "box": r["box"],
                    "log_speedup": float(d.mean()), "low": float(lo), "high": float(hi),
                    "generated_faster": bool(lo > 0), "generated_periodic": r.get("generated_periodic")})
        allp.append(d)
    spread = {}
    for fac in ("layer-heis", "layer"):
        rs = [r for r in rows if r["factor"] == fac and r.get("generated_ns_per_update")]
        if rs:
            xm = [float(np.median(r["xla_ns_per_update"])) for r in rs]
            gm = [float(np.median(r["generated_ns_per_update"])) for r in rs]
            spread[fac] = {"xla_max_over_min": max(xm) / min(xm), "generated_max_over_min": max(gm) / min(gm),
                           "L1": [r["box"][0] for r in rs], "xla_median_ns": xm, "generated_median_ns": gm}
    return {"grids": per, "spread": spread, "all_faster": bool(per) and all(p["generated_faster"] for p in per),
            "min_log_speedup": min((p["log_speedup"] for p in per), default=None),
            "max_log_speedup": max((p["log_speedup"] for p in per), default=None)}


def conf_verdict(rows, threshold: int, boots: int = 2000, seed: int = 0):
    """The v04 E5 criteria. (a) every grid at or above tC logs the factor (11) among its factors and no grid
    below logs a factor above 1; (b) the pooled Heisenberg log ratio of u = 4 against its odd neighbours has
    a 95% interval excluding zero and |estimate| >= log 1.3; a device is inconclusive for (b) when the
    median SM clock of its u = 4 and u = 1 grids differs by more than 5%."""
    import numpy as np
    a = []
    for r in rows:
        if r["factor"] not in ("below", "above"):
            continue
        logged = r.get("logged_unroll") or []
        pred = rule(r["elements"], threshold)
        ok = (pred in logged) if r["elements"] >= threshold else (bool(logged) and max(logged) <= 1)
        a.append({"factor": r["factor"], "n": r["n"], "b": r["b"], "box": r["box"], "elements": r["elements"],
                  "predicted": pred, "logged": sorted(set(logged)), "holds": bool(ok)})
    heis = h4_cost(rows, "layer-heis", boots, seed)
    ising = h4_cost(rows, "layer", boots, seed)

    def clock(us):
        v = [c for r in rows if r["factor"] == "layer-heis" and (r["box"][0] % 4 == 0) == (us == 4)
             for c in r.get("sm_clock_mhz", []) if c]
        return float(np.median(v)) if v else None
    c4, c1 = clock(4), clock(1)
    p = heis["pooled"]
    b_holds = bool(p) and (p["low"] > 0 or p["high"] < 0) and abs(p["log_ratio"]) >= MARGIN_H4
    inconclusive = bool(c4 and c1) and abs(c4 / c1 - 1) > 0.05
    return {"a": {"grids": a, "holds": bool(a) and all(x["holds"] for x in a)},
            "b": {"heisenberg": heis, "ising_reported": ising, "holds": b_holds, "margin": MARGIN_H4,
                  "median_sm_clock_u4": c4, "median_sm_clock_u1": c1, "inconclusive": inconclusive,
                  "sign": None if not p else ("u4 slower" if p["log_ratio"] > 0 else "u4 faster")}}


def grids(quick: bool):
    if quick:
        return [("B", B, 3, 1, (15, 15, 15)) for B in (1, 4, 5, 8)] + [("threshold", 4, 1, 1, (L1, 15, 15))
                                                                       for L1 in (40, 48)]
    g = [("B", B, 3, 1, (15, 15, 15)) for B in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)]
    g += [("n", 13, n, 1, (15, 15, 15)) for n in (1, 2, 3)]
    g += [("b", 13, 1, b, (15, 15, 15)) for b in (1, 2, 4)]
    g += [("layer", 13, 1, 1, (L1, 15, 15)) for L1 in (15, 16, 17, 18, 19, 20, 22, 24)]
    g += [("threshold", 4, 1, 1, (L1, 15, 15)) for L1 in (36, 40, 44, 48, 52, 56)]
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="E5b")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--set", choices=("b", "c", "conf", "gen", "genb"), default="b",
                    help="b: grids around the threshold; c: N >= 10^6; conf: experiments v04 E5 confirmatory grids")
    a = ap.parse_args(argv)
    conf = a.set == "conf"
    genset = a.set in ("gen", "genb")
    if conf or genset:   # the device threshold decides the (a) grids, so read it before JAX starts
        import torch
        props0 = torch.cuda.get_device_properties(0)
        threshold0 = props0.max_threads_per_multi_processor * props0.multi_processor_count
    dump = Path(tempfile.mkdtemp(prefix="xla_dump_b_"))
    logf = Path(tempfile.mkdtemp(prefix="xla_log_b_")) / "stderr.log"
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={dump} --xla_dump_hlo_as_text"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["TF_CPP_VMODULE"] = "gpu_fusible=2"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # C++ logging writes to file descriptor 2: send it to a file, keep Python's own prints on stdout
    global _SAVED, _LOGF
    saved = _SAVED = os.dup(2)
    _LOGF = logf
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 2)

    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Spin_Colour_Kernel import colouring as co
    from Spin_Colour_Kernel import crystals as cr

    crystal_for_b = {1: "sc", 2: "bcc", 4: "fcc"}

    make_step = make_baseline_step          # the E5 baseline, unchanged

    G = (grids_conf(threshold0, a.quick) if conf else grids_genb(a.quick) if a.set == "genb"
         else grids_gen(threshold0, a.quick) if genset
         else grids_c(a.quick) if a.set == "c" else grids(a.quick))
    mon = None
    if conf or genset:   # SM clock and power during every timing (the v04 inconclusive rule reads the clock)
        from Spin_Colour_Kernel.exp_perf import GpuMonitor
        mon = GpuMonitor("cuda:0")
        time.sleep(1.0)
    rows = []
    compiled = []      # (step, state, key, times) per grid; timed after compilation, interleaved over grids
    t0 = time.time()
    for gid, (factor, B, n, b, box) in enumerate(G):
        L1, L2, L3 = box
        shutil.rmtree(dump, ignore_errors=True)
        dump.mkdir(parents=True, exist_ok=True)
        os.write(2, f"\n{MARK} {gid}\n".encode())
        f = make_step(n, b, box, gid)
        S = jnp.ones((B, n, b, L3, L2, L1), jnp.float32) / np.sqrt(n)
        key = jax.random.PRNGKey(gid)
        S2, key = f(S, key, 0.5)
        S2.block_until_ready()
        times = []
        compiled.append((f, S2, key, times))
        fusions = []
        hlo = [p for p in glob.glob(str(dump / "**" / "*after_optimizations*.txt"), recursive=True)
               if f"wlb{gid:04d}" in os.path.basename(p) and "buffer" not in os.path.basename(p)]
        for p in hlo:
            for dt, dims, kind in FUSION.findall(Path(p).read_text(errors="ignore")):
                shape = [int(d) for d in dims.split(",") if d]
                fusions.append({"dtype": dt, "shape": shape, "elements": int(np.prod(shape)) if shape else 1,
                                "kind": kind})
        sites = B * b * L1 * L2 * L3
        rows.append({"factor": factor, "B": B, "n": n, "b": b, "box": list(box), "elements": B * n * b * L1 * L2 * L3,
                     "sites": sites, "xla_seconds_per_sweep": times,
                     "fusions": fusions, "hlo_files": len(hlo)})
        print(f"E5b grid {gid:2d} {factor:9s} B={B} n={n} b={b} box={box} N={rows[-1]['elements']} "
              f"fusions={len(fusions)} compiled {time.time() - t0:.0f}s", flush=True)
    os.write(2, f"\n{MARK} end\n".encode())
    # timing: every grid once per round, order reversed on odd rounds, so a drift of the device over the
    # session cannot line up with the grid order (the T4 runs drifted with its 70 W power cap)
    nit = [20] * len(compiled)
    gens = [None] * len(compiled)
    if genset:   # the generated kernel on the same model, crystal and box, timed in the same rounds
        import torch
        from Spin_Colour_Kernel import system as sy
        from Spin_Colour_Kernel.sampler import Sampler
        for i, (factor, B, n, b, box) in enumerate(G):
            c = cr.make(crystal_for_b[b])
            for per_ in ((True,) * 3, (False,) * 3):
                try:
                    s_ = sy.System(f"E5gen {crystal_for_b[b]} n={n}", n, c, box, per_, {1: 1.0})
                    break
                except ValueError:
                    s_ = None
            A = Sampler(s_, B, "cuda:0", seed=1)
            A.set_temperature(2.0)
            X = A.init_spins(seed=1)
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            rep_ = A.graph(X, 10)
            rep_()
            torch.cuda.synchronize()
            ev[0].record(); rep_(); ev[1].record(); torch.cuda.synchronize()
            k = max(1, math.ceil(0.1 / (ev[0].elapsed_time(ev[1]) / 1e3)))   # >= ~100 ms per measurement
            gens[i] = {"replay": rep_, "k": k, "sites": s_.n_active * B, "periodic": s_.periodic[0], "ns": []}
    if conf or genset:   # >= ~100 ms per measurement, so the 10 ms clock sampler sees each one several times
        for i, (f, S2, key, times) in enumerate(compiled):
            t = time.perf_counter()
            for _ in range(3):
                S2, key = f(S2, key, 0.5)
            S2.block_until_ready()
            nit[i] = max(20, math.ceil(0.1 / ((time.perf_counter() - t) / 3)))
            compiled[i] = (f, S2, key, times)
    for r_ in range(a.reps):
        order = range(len(compiled)) if r_ % 2 == 0 else reversed(range(len(compiled)))
        for i in order:
            f, S2, key, times = compiled[i]
            w0 = time.time()
            t = time.perf_counter()
            for _ in range(nit[i]):
                S2, key = f(S2, key, 0.5)
            S2.block_until_ready()
            times.append((time.perf_counter() - t) / nit[i])
            compiled[i] = (f, S2, key, times)
            if gens[i] is not None:
                gi = gens[i]
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(gi["k"]):
                    gi["replay"]()
                torch.cuda.synchronize()
                gi["ns"].append((time.perf_counter() - t) * 1e9 / (gi["k"] * 10) / gi["sites"])
            if mon is not None:
                g = mon.window(w0, time.time())
                rows[i].setdefault("sm_clock_mhz", []).append(g.get("clocks.sm"))
                rows[i].setdefault("power_w", []).append(g.get("power.draw"))
                rows[i].setdefault("throttle", []).append(g.get("throttle_reasons"))
    for row in rows:
        row["xla_ns_per_update"] = [t * 1e9 / row["sites"] for t in row["xla_seconds_per_sweep"]]
        print(f"E5b {row['factor']:9s} B={row['B']} n={row['n']} b={row['b']} box={tuple(row['box'])} "
              f"N={row['elements']} ns/upd={np.median(row['xla_ns_per_update']):.3f}", flush=True)
    os.dup2(saved, 2)
    os.close(fd)

    # split the C++ log at the grid marks and read the logged unroll factors
    log = logf.read_bytes()
    parts = re.split(rb"\n" + MARK.encode() + rb" (\d+|end)\n", log)
    per = {}
    for i in range(1, len(parts) - 1, 2):
        if parts[i] != b"end":
            per[int(parts[i])] = [int(x) for x in UNROLL.findall(parts[i + 1])]
    vlog_seen = sum(len(v) for v in per.values())

    # the generated kernel on the same model, crystal and box
    import torch
    from Spin_Colour_Kernel import system as sy
    from Spin_Colour_Kernel.sampler import Sampler
    props = torch.cuda.get_device_properties(0)
    threshold = props.max_threads_per_multi_processor * props.multi_processor_count
    for r in rows:
        B, n, b, box = r["B"], r["n"], r["b"], tuple(r["box"])
        c = cr.make(crystal_for_b[b])
        s = None
        for per_ in ((True,) * 3, (False,) * 3):
            try:
                s = sy.System(f"E5b {crystal_for_b[b]} n={n}", n, c, box, per_, {1: 1.0})
                break
            except ValueError:
                continue
        gen = []
        if genset:
            gen = gens[rows.index(r)]["ns"]
            r["generated_periodic"] = gens[rows.index(r)]["periodic"]
        elif s is not None and not conf:   # v04 E5 tests the baseline only
            A = Sampler(s, B, "cuda:0", seed=1)
            A.set_temperature(2.0)
            X = A.init_spins(seed=1)
            rep = A.graph(X, 10)
            rep()
            torch.cuda.synchronize()
            for _ in range(a.reps):
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                rep()
                ev1.record()
                torch.cuda.synchronize()
                gen.append(ev0.elapsed_time(ev1) * 1e6 / 10 / s.n_active / B)
            r["generated_periodic"] = s.periodic[0]
        r["generated_ns_per_update"] = gen
        loops = [fu for fu in r["fusions"] if fu["kind"] == "kLoop"]
        for fu in loops:
            fu["predicted_unroll"] = rule(fu["elements"], threshold, BITS.get(fu["dtype"], 32))
        r["predicted_unroll"] = sorted(fu["predicted_unroll"] for fu in loops)
        r["logged_unroll"] = sorted(per.get(rows.index(r), []))
        r["multiset_match"] = (r["logged_unroll"] == r["predicted_unroll"]) if r["logged_unroll"] else None
        big = max(loops, key=lambda fu: fu["elements"], default=None)
        r["largest_loop_fusion"] = big
        r["eq11_no_threshold"] = math.gcd(r["elements"], 4)
        r["eq11_threshold"] = rule(r["elements"], threshold)
        r["logged_max"] = max(r["logged_unroll"], default=None)
        print(f"E5b {r['factor']:9s} B={B} n={n} b={b} box={box} logged={Counter(r['logged_unroll'])} "
              f"pred={Counter(r['predicted_unroll'])} gen ns/upd={np.median(gen) if gen else float('nan'):.3f}",
              flush=True)

    if mon is not None:
        mon.close()
    summary = {"threshold": threshold, "device": props.name, "grids": len(rows), "vlog_lines": vlog_seen,
               "multiset_matches": sum(bool(r["multiset_match"]) for r in rows),
               "multiset_readable": sum(r["multiset_match"] is not None for r in rows),
               "max_matches_eq11_threshold": sum(r["logged_max"] == r["eq11_threshold"] for r in rows),
               "max_matches_eq11_no_threshold": sum(r["logged_max"] == r["eq11_no_threshold"] for r in rows)}
    if a.set == "genb":
        summary["v16"] = v6 = genb_verdict(rows)
        print(f"{a.tag} generated smooth: {v6['generated_smooth']}; faster everywhere: {v6['generated_faster_everywhere']}; "
              f"jumps xla/gen { {k: (x['xla_upward_jumps'], x['generated_upward_jumps']) for k, x in v6['per_model'].items()} }",
              flush=True)
    elif genset:
        summary["v15"] = gen_verdict(rows)
        v = summary["v15"]
        print(f"{a.tag} generated faster on all {len(v['grids'])} grids: {v['all_faster']}; log speed-up "
              f"{v['min_log_speedup']:.3f} to {v['max_log_speedup']:.3f}; spread "
              f"{ {k: (round(x['xla_max_over_min'], 3), round(x['generated_max_over_min'], 3)) for k, x in v['spread'].items()} }",
              flush=True)
    if conf:
        summary["v04"] = conf_verdict(rows, threshold)
        v = summary["v04"]
        print(f"{a.tag} (a) holds={v['a']['holds']} on {len(v['a']['grids'])} grids; (b) pooled="
              f"{v['b']['heisenberg']['pooled']} holds={v['b']['holds']} inconclusive={v['b']['inconclusive']} "
              f"clock u4/u1={v['b']['median_sm_clock_u4']}/{v['b']['median_sm_clock_u1']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    import platform
    payload = {"experiment": a.tag, "rows": rows, "summary": summary,
               "environment": {"python": platform.python_version(), "jax": jax.__version__,
                               "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "devices": [str(d) for d in jax.devices()]},
               "log_sample": log[:4000].decode(errors="ignore") if not vlog_seen else None,
               "source": {"gpu_fusible.cc": "ComputeLoopFusionConfig, openxla/xla main, read 2026-11-06",
                          "launch_dimensions.cc": "CalculateLaunchDimensions, openxla/xla main, read 2026-11-06"}}
    try:
        import jaxlib
        payload["environment"]["jaxlib"] = jaxlib.__version__
    except Exception:
        pass
    (a.out / f"{a.tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    shutil.rmtree(dump, ignore_errors=True)
    print(a.tag, "summary", json.dumps(summary), flush=True)


_SAVED, _LOGF = None, None


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # stderr points at the XLA log while the grids run: restore it and show the log's tail
        if _SAVED is not None:
            os.dup2(_SAVED, 2)
        if _LOGF is not None and Path(_LOGF).exists():
            print(Path(_LOGF).read_text(errors="ignore")[-4000:], flush=True)
        raise
