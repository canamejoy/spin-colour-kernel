"""Experiments E4, E6, E7 and the supplementary two-GPU measurement E8 of experiments v03."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import system as sy
from .exp_core import integrated_tau, save
from .sampler import Sampler

ISING_GPU_COMMIT = "935796d7a26016670363af7a0dced8a9ebcd4714"


def _events_ms(fn, reps: int, device) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize(device)
    return a.elapsed_time(b)


def boot_ci(x: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(x)), float(lo), float(hi)


# ------------------------------------------------------------------ rate of the generated kernel
def rate(s: sy.System, B: int, device="cuda", T: float | None = None, target_updates: float = 2e7,
         reps: int = 10, seed: int = 1, compact: bool = True, packed: bool | None = None):
    """Spin updates per ns over `reps` timed CUDA-graph replays; returns per-replay rates."""
    A = Sampler(s, B, device, seed, compact=compact, packed=packed)
    A.set_temperature(T or s.T_ref or 1.0)
    S = A.init_spins(seed=seed)
    per_sweep = B * s.n_active
    k = max(1, int(math.ceil(target_updates / per_sweep)))
    replay = A.graph(S, k)
    for _ in range(2):
        replay()
    out = []
    for _ in range(reps):
        ms = _events_ms(replay, 1, A.device)
        out.append(per_sweep * k / (ms * 1e6))
    return out, {"sweeps_per_replay": k, "updates_per_sweep": per_sweep}


def throughput_systems() -> list[sy.System]:
    return [
        sy.ising("square", 256, T_ref=sy.TC["Ising square"]),
        sy.ising("triangular", 192, T_ref=sy.TC["Ising triangular"]),
        sy.ising("sc", 32, T_ref=sy.TC["Ising sc"]),
        sy.ising("bcc", 32, T_ref=sy.TC["Ising bcc"]),
        sy.ising("fcc", 32, T_ref=sy.TC["Ising fcc"]),
        sy.xy("square", 256, T_ref=0.893),
        sy.xy("sc", 32, T_ref=sy.TC["XY sc"]),
        sy.heisenberg("sc", 32, T_ref=sy.TC["Heisenberg sc"]),
        sy.heisenberg("bcc", 32, T_ref=sy.TC["Heisenberg bcc"]),
        sy.heisenberg("fcc", 32, T_ref=1.0),
        sy.heisenberg("triangular", 192, J=-1.0, T_ref=0.285, label="Heisenberg triangular AFM"),
        sy.heisenberg("kagome", 192, J=-1.0, T_ref=0.1, label="Heisenberg kagome AFM"),
        sy.nanodot(),
    ]


def _split_args(text: str, start: int):
    """Top-level arguments of the call whose '(' is at text[start]; returns (args, end index)."""
    depth, args, cur = 0, [], []
    i = start
    while True:
        ch = text[i]
        if ch == "(":
            depth += 1
            if depth > 1:
                cur.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i
            cur.append(ch)
        elif ch == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1


def patch_cuda13(src: str) -> str:
    """CUDA 13 takes a cudaMemLocation in cudaMemAdvise and cudaMemPrefetchAsync (and a flags argument)."""
    out, i = [], 0
    for name in ("cudaMemAdvise", "cudaMemPrefetchAsync"):
        out, i = [], 0
        while True:
            j = src.find(name + "(", i)
            if j < 0:
                out.append(src[i:])
                break
            k = j + len(name)
            args, end = _split_args(src, k)
            out.append(src[i:j])
            if name == "cudaMemAdvise" and len(args) == 4:
                args[3] = f"cudaMemLocation{{cudaMemLocationTypeDevice, (int)({args[3]})}}"
            elif name == "cudaMemPrefetchAsync" and len(args) == 4:
                args = args[:2] + [f"cudaMemLocation{{cudaMemLocationTypeDevice, (int)({args[2]})}}", "0", args[3]]
            out.append(name + "(" + ", ".join(args) + ")")
            i = end + 1
        src = "".join(out)
    return src


def build_ising_gpu(work: Path, sm: str) -> Path | None:
    """Clone NVIDIA ising-gpu at a pinned commit and build the optimized multi-spin code."""
    d = work / "ising-gpu"
    exe = d / "optimized" / "cuIsingModel" / "cuIsing"
    if exe.exists():
        return exe
    try:
        if not d.exists():
            subprocess.run(["git", "clone", "-q", "https://github.com/NVIDIA/ising-gpu", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", ISING_GPU_COMMIT], check=True)
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        mk = ["make", "-C", str(d / "optimized" / "cuIsingModel"), f"CUDA_HOME={cuda}", f"SMS={sm}"]
        r = subprocess.run(mk, capture_output=True, text=True)
        if r.returncode:
            # CUDA >= 13 changed two managed-memory signatures; patch the call sites and rebuild
            main = d / "optimized" / "cuIsingModel" / "main.cu"
            main.write_text(patch_cuda13(main.read_text()))
            subprocess.run(mk, check=True, capture_output=True, text=True)
    except Exception as e:  # reported, never hidden
        print("ising-gpu build failed:", repr(e), getattr(e, "stderr", "")[-2000:] if hasattr(e, "stderr") else "")
        return None
    return exe if exe.exists() else None


CUISING2025_SHA256 = "0be8ebb2a0d57cd60aeaffc494d94d81b49d6eefe2470f4882948365e07871bd"


def build_cuising2025(work: Path, sm: str) -> Path | None:
    """Build cuIsing (optimized) of Bisson et al., CPC 315 (2025) 109690, from the vendored, checksummed archive
    (third_party/cuIM_BC-2025.tar.bz2, MIT). Its CLI and the 'spins/ns' line match ising-gpu's."""
    import hashlib
    import tarfile
    arc = Path(__file__).resolve().parents[2] / "third_party" / "cuIM_BC-2025.tar.bz2"
    if not arc.exists() or hashlib.sha256(arc.read_bytes()).hexdigest() != CUISING2025_SHA256:
        print("cuIsing 2025: archive missing or checksum mismatch", flush=True)
        return None
    d = work / "cuIM_BC-2025"
    exe = d / "cuIM_BC" / "cuIsingModel" / "cuIsing"
    if exe.exists():
        return exe
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arc) as t:
        t.extractall(d)
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    r = subprocess.run(["make", "-C", str(d / "cuIM_BC" / "cuIsingModel"), f"CUDA_HOME={cuda}", f"SMS={sm}"],
                       capture_output=True, text=True)
    if r.returncode or not exe.exists():
        print("cuIsing 2025 build failed:", (r.stdout + r.stderr)[-2000:], flush=True)
        return None
    return exe


def kshape(out: Path, device="cuda", L: int = 4096, B: int = 64, seed: int = 77, tag: str = "Kshape",
           shapes=((1, 32, 1), (1, 32, 2), (2, 32, 1), (2, 32, 2), (1, 64, 2), (2, 64, 2), (4, 32, 2), (4, 64, 2),
                   (4, 128, 2), (4, 256, 2), (8, 64, 2))):
    """Diagnostic (close-reviewer-gaps, cycle 1): independent bitwise rate at one large lattice for small and large
    launch shapes (words, lanes, warps) per program; not a registered test."""
    s = sy.ising("square", L, T_ref=sy.TC["Ising square"])
    res = {}
    for G, tile, nw in shapes:
        try:
            A = Sampler(s, B, device, seed, bitwise=True, independent=True, words_per_program=G, bitwise_tile=tile,
                        bitwise_warps=nw)
            A.set_temperature(sy.TC["Ising square"])
            S = A.init_spins(seed=seed)
            rp = A.graph(S, 2)
            rp()
            res[f"{G},{tile},{nw}"] = float(np.median([B * L * L * 2 / (_events_ms(rp, 1, A.device) * 1e6) for _ in range(5)]))
            print(f"Kshape {G},{tile},{nw}: {res[f'{G},{tile},{nw}']:.0f}/ns", flush=True)
            del A, S, rp
        except Exception as exc:
            res[f"{G},{tile},{nw}"] = repr(exc)[:200]
            print("Kshape", G, tile, nw, repr(exc)[:150], flush=True)
        torch.cuda.empty_cache()
    return save(out, tag, {"L": L, "B": B, "rates": res})


def e6_scan(out: Path, device="cuda", sides=(4096, 8192, 16384, 32768), reps: int = 10, seed: int = 5011,
            work: Path = Path("/tmp/sck"), tag: str = "E6scan", warmup: int = 0, rounds=(7, 10),
            verdict_rounds: int = 7):
    """Experiments v11, H5(b) of proposal r14: size scan against ising-gpu and cuIsing 2025.

    Specialists at side^2 spins; ours (independent bitwise, 64 replicas of (side/8)^2, equal spins per sweep) with
    7 and with 10 Philox rounds. Our launch shape is fixed once by a pilot at the first side (larger sides held
    out). Per side the four codes alternate `reps` rounds (order reversed every round). The verdict per device uses
    the faster specialist at its fastest side."""
    from . import bitwise as bw
    sm = _sm()
    exes = {"ising-gpu": build_ising_gpu(work, sm), "cuIsing-2025": build_cuising2025(work, sm)}
    exes = {k: v for k, v in exes.items() if v is not None}
    B = 64

    def mk(L, rounds, G, tile, nw):
        old = bw.INDEPENDENT_ROUNDS
        bw.INDEPENDENT_ROUNDS = rounds
        try:
            s = sy.ising("square", L, T_ref=sy.TC["Ising square"])
            A = Sampler(s, B, device, seed, bitwise=True, independent=True, words_per_program=G,
                        bitwise_tile=tile, bitwise_warps=nw)
        finally:
            bw.INDEPENDENT_ROUNDS = old
        A.set_temperature(sy.TC["Ising square"])
        S = A.init_spins(seed=seed)
        k = max(2, int(2e10 // (B * L * L)) // 10)
        rp = A.graph(S, k)
        rp()
        return A, S, rp, B * L * L * k

    pilot = {}
    L0 = sides[0] // 8
    for G in (4, 8, 16):
        for tile in (64, 128, 256):
            for nw in (2, 4, 8):
                try:
                    A, S, rp, upd = mk(L0, 7, G, tile, nw)
                    pilot[f"{G},{tile},{nw}"] = float(np.median([upd / (_events_ms(rp, 1, A.device) * 1e6) for _ in range(3)]))
                    del A, S, rp
                except Exception as exc:
                    pilot[f"{G},{tile},{nw}"] = 0.0
                    print("pilot", G, tile, nw, repr(exc)[:120], flush=True)
                torch.cuda.empty_cache()
    shape = max(pilot, key=pilot.get)
    G, tile, nw = (int(x) for x in shape.split(","))
    rows = []
    for side in sides:
        L = side // 8
        try:
            ours = {r: mk(L, r, G, tile, nw) for r in rounds}
        except Exception as exc:          # a side our batch cannot hold on this device
            rows.append({"side": side, "error": repr(exc)[:300]})
            torch.cuda.empty_cache()
            continue
        steps = max(10, int(2e10 // (side * side)))
        rate = {f"ours-r{r}": [] for r in ours} | {n: [] for n in exes}
        clocks = []
        mon = GpuMonitor(device)
        for i in range(-warmup, reps):           # negative rounds are warm-up, discarded (v12)
            order = list(rate) if i % 2 == 0 else list(rate)[::-1]
            for n in order:
                t0 = time.time()
                if n.startswith("ours"):
                    A, S, rp, upd = ours[int(n[6:])]
                    v = upd / (_events_ms(rp, 1, A.device) * 1e6)
                else:
                    try:
                        v = run_ising_gpu(exes[n], side, side, steps)
                    except Exception as exc:
                        v = float("nan")
                        print("E6scan", n, side, repr(exc)[:160], flush=True)
                if i >= 0:
                    rate[n].append(v)
                    clocks.append({"code": n, "round": i, **mon.window(t0, time.time())})
        mon.close()
        row = {"side": side, "spins": side * side, "ours_lattice": f"{B} x {L}^2", "rates": rate, "clocks": clocks,
               "warmup_rounds_discarded": warmup}
        for n in exes:
            v = np.asarray(rate[n])
            if np.all(np.isfinite(v)):
                for r in rounds:
                    m, lo, hi = boot_ci(np.log(np.asarray(rate[f"ours-r{r}"]) / v))
                    row[f"r{r}_vs_{n}"] = {"mean": m, "low": lo, "high": hi}
        rows.append(row)
        print(f"E6scan {side}^2: " + "  ".join(f"{n} {np.nanmean(v):.0f}/ns" for n, v in rate.items()), flush=True)
        del ours
        torch.cuda.empty_cache()
    # H5(b): the faster specialist at its fastest side
    best = None
    for row in rows:
        for n in exes:
            v = row.get("rates", {}).get(n)
            if v and np.all(np.isfinite(v)) and (best is None or np.mean(v) > best[2]):
                best = (row["side"], n, float(np.mean(v)))
    verdict = None
    if best:
        row = next(r for r in rows if r["side"] == best[0])
        ci = row[f"r{verdict_rounds}_vs_{best[1]}"]
        verdict = {"specialist": best[1], "side": best[0], "specialist_rate": best[2], "log_ratio": ci,
                   "margin": -math.log(1.05), "non_inferior": bool(ci["low"] > -math.log(1.05)),
                   "verdict_rounds": verdict_rounds,
                   "other_rounds": {r: row[f"r{r}_vs_{best[1]}"] for r in rounds if r != verdict_rounds}}
        print(f"E6scan H5(b): best specialist {best[1]} at {best[0]}^2 ({best[2]:.0f}/ns); ours r{verdict_rounds} log ratio "
              f"{ci['mean']:+.3f} [{ci['low']:+.3f},{ci['high']:+.3f}] non-inferior={verdict['non_inferior']}", flush=True)
    return save(out, tag, {"pilot": pilot, "launch_shape": shape, "rows": rows, "verdict": verdict, "seed": seed,
                           "ising_gpu_commit": ISING_GPU_COMMIT, "cuising2025_sha256": CUISING2025_SHA256})


def e6_specialists(out: Path, device="cuda", reps: int = 10, seed: int = 4009, work: Path = Path("/tmp/sck"),
                   tag: str = "E6spec"):
    """Experiments v09, E6: the independent bitwise mode against ising-gpu (2020) and cuIsing 2025 (Bisson et al.),
    2D square Ising at Tc, 4096^2 spins per sweep on every side (ours: 64 x 512^2), the three codes alternated
    on the same device for `reps` rounds; paired log speed ratios with bootstrap intervals."""
    sm = _sm()
    exes = {"ising-gpu": build_ising_gpu(work, sm), "cuIsing-2025": build_cuising2025(work, sm)}
    s = sy.ising("square", 512, T_ref=sy.TC["Ising square"])
    B, k = 64, 50
    best, pilot = None, {}
    # pilot over the launch shape (words per program, lanes per program, warps), fastest median of 3 replays
    for G in (4, 8, 16):
        for tile in (64, 128, 256):
            for nw in (2, 4, 8):
                try:
                    A = Sampler(s, B, device, seed, bitwise=True, independent=True, words_per_program=G,
                                bitwise_tile=tile, bitwise_warps=nw)
                    A.set_temperature(sy.TC["Ising square"])
                    S = A.init_spins(seed=seed)
                    rp = A.graph(S, k)
                    rp()
                    pilot[f"{G},{tile},{nw}"] = float(np.median(
                        [B * s.n_active * k / (_events_ms(rp, 1, A.device) * 1e6) for _ in range(3)]))
                    del A, S, rp
                except Exception as exc:
                    pilot[f"{G},{tile},{nw}"] = 0.0
                    print("pilot", G, tile, nw, repr(exc)[:120], flush=True)
                torch.cuda.empty_cache()
    best = max(pilot, key=pilot.get)
    bg, bt, bn = (int(x) for x in best.split(","))
    A = Sampler(s, B, device, seed, bitwise=True, independent=True, words_per_program=bg, bitwise_tile=bt,
                bitwise_warps=bn)
    A.set_temperature(sy.TC["Ising square"])
    S = A.init_spins(seed=seed)
    rp = A.graph(S, k)
    rp()
    ours, base = [], {n: [] for n, e in exes.items() if e is not None}
    for i in range(reps):
        ours.append(B * s.n_active * k / (_events_ms(rp, 1, A.device) * 1e6))
        for n in base:
            base[n].append(run_ising_gpu(exes[n], 4096, 4096, 200))
    res = {"independent_flips_per_ns": ours, "words_per_program": best, "pilot": pilot, "seed": seed,
           "lattice": "64 x 512^2 (independent bitwise) vs 4096^2", "ising_gpu_commit": ISING_GPU_COMMIT,
           "cuising2025_sha256": CUISING2025_SHA256, "built": {n: e is not None for n, e in exes.items()}}
    for n, v in base.items():
        m, lo, hi = boot_ci(np.log(np.asarray(ours) / np.asarray(v)))
        res[n] = {"spins_per_ns": v, "log_ratio": m, "low": lo, "high": hi}
        print(f"E6spec ours {np.mean(ours):.0f}/ns vs {n} {np.mean(v):.0f}/ns  log ratio {m:+.3f} [{lo:+.3f},{hi:+.3f}]",
              flush=True)
    return save(out, tag, res)


def run_ising_gpu(exe: Path, x: int, y: int, steps: int, gpus: int = 1) -> float:
    r = subprocess.run([str(exe), "-x", str(x), "-y", str(y), "-n", str(steps), "-a", "1.0", "-g", str(gpus)],
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r"spins/ns:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError("ising-gpu output without spins/ns: " + r.stdout[-1500:] + r.stderr[-1500:])
    return float(m.group(1))


def _sm() -> str:
    maj, mi = torch.cuda.get_device_capability(0)
    return f"{maj}{mi}"


# ------------------------------------------------------------------ E6
_GPU_FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")


def gpu_state(device="cuda:0") -> dict:
    """SM/memory clock (MHz), temperature (C), power (W) and active throttle reasons, via nvidia-smi.

    Read outside the timed region; an empty dict when nvidia-smi is unavailable."""
    idx = str(torch.device(device).index or 0)
    for reasons in ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None):
        q = ",".join(_GPU_FIELDS + ((reasons,) if reasons else ()))
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", idx],
                               capture_output=True, text=True, timeout=10)
        except Exception:
            return {}
        if r.returncode == 0 and r.stdout.strip():
            vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
            out = {}
            for k, v in zip(_GPU_FIELDS, vals):
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = None
            if reasons:
                out["throttle_reasons"] = vals[-1]
            return out
    return {}


class GpuMonitor:
    """`nvidia-smi -lms` in the background: SM clock, temperature, power and throttle reasons every `ms`.

    `window(t0, t1)` summarises the samples taken between two `time.time()` stamps, so a measurement is
    attributed the clock the GPU actually ran at while it was busy, not the idle clock read before it."""

    def __init__(self, device="cuda:0", ms: int = 10):
        self.samples: list[tuple[float, float, float, float, str]] = []
        self.proc = None
        idx = str(torch.device(device).index or 0)
        for reasons in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
            q = f"timestamp,clocks.sm,temperature.gpu,power.draw,{reasons}"
            try:
                probe = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", idx],
                                       capture_output=True, text=True, timeout=10)
            except Exception:
                return
            if probe.returncode == 0:
                self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
                                              "-i", idx, f"-lms={ms}"], stdout=subprocess.PIPE, text=True)
                import threading
                threading.Thread(target=self._read, daemon=True).start()
                return

    def _read(self):
        from datetime import datetime
        for line in self.proc.stdout:
            v = [x.strip() for x in line.split(",")]
            try:
                ts = datetime.strptime(v[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((ts, float(v[1]), float(v[2]), float(v[3]), v[4]))
            except (ValueError, IndexError):
                continue

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in list(self.samples) if t0 <= x[0] <= t1]
        if not s:
            return {"n_samples": 0}
        sm = [x[1] for x in s]
        return {"n_samples": len(s), "clocks.sm": float(np.mean(sm)), "clocks.sm_min": min(sm),
                "clocks.sm_max": max(sm), "temperature.gpu": max(x[2] for x in s),
                "power.draw": float(np.mean([x[3] for x in s])), "throttle_reasons": sorted({x[4] for x in s})}

    def close(self):
        if self.proc is not None:
            self.proc.terminate()


def _slope_ci(data: dict, reps: int, seed: int = 0):
    """Slope of log cost per update against log2 batch, with a bootstrap 95% interval over repetitions."""
    Bs = np.array(sorted(data))
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(2000):
        y = [np.log(1.0 / np.mean(np.asarray(data[B])[rng.integers(0, reps, reps)])) for B in Bs]
        slopes.append(np.polyfit(np.log2(Bs), y, 1)[0])
    slope = float(np.polyfit(np.log2(Bs), [np.log(1.0 / np.mean(data[B])) for B in Bs], 1)[0])
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return slope, float(lo), float(hi)


MARGIN_H5 = math.log(1.05)   # experiments v04, E6: upper end of the slope interval per batch doubling


def e6(out: Path, device="cuda", batches=(8, 16, 32, 64, 128), reps: int = 10, work: Path = Path("/tmp/sck"),
       seed: int = 1, tag: str = "E6"):
    """Flat cost in batch size per system (H5) and the paired comparison with ising-gpu on 2D Ising.

    Batch sizes are interleaved: every round times one replay of each batch size, in an order that
    reverses from round to round, so slow drift of the clock (a passively cooled T4 heats up over a run)
    lands on every batch size alike instead of on the last ones. The SM clock, temperature, power and
    throttle reasons are sampled by a background nvidia-smi during every measurement (>= ~100 ms of
    replays); a second slope uses the rate per mean SM MHz."""
    slope_rows = []
    mon = GpuMonitor(device)
    time.sleep(1.0)
    for s in throughput_systems():
        prep = {}
        for B in batches:
            if B * s.n_active * s.n * 4 > 2.5e9:
                continue
            A = Sampler(s, B, device, seed)
            A.set_temperature(s.T_ref or 1.0)
            S = A.init_spins(seed=seed)
            per_sweep = B * s.n_active
            k = max(1, int(math.ceil(2e7 / per_sweep)))
            replay = A.graph(S, k)
            for _ in range(2):
                replay()
            ms0 = _events_ms(replay, 1, A.device)
            nrep = max(1, int(math.ceil(100.0 / ms0)))     # >= ~100 ms per measurement: several clock samples
            prep[B] = (replay, per_sweep * k * nrep, A.device, nrep)
        order = sorted(prep)
        data = {B: [] for B in order}
        norm = {B: [] for B in order}
        states = {B: [] for B in order}
        for i in range(reps):
            for B in (order if i % 2 == 0 else order[::-1]):
                replay, upd, dev, nrep = prep[B]
                t0 = time.time()
                ms = _events_ms(replay, nrep, dev)
                g = mon.window(t0, time.time())
                r = upd / (ms * 1e6)
                data[B].append(r)
                states[B].append(g)
                if g.get("clocks.sm"):
                    norm[B].append(r / g["clocks.sm"])
        del prep
        torch.cuda.empty_cache()
        Bs = np.array(order)
        slope, lo, hi = _slope_ci(data, reps)
        row = {"system": s.name, "active_sites": s.n_active, "batches": Bs.tolist(),
               "flips_per_ns": {int(B): data[B] for B in Bs},
               "gpu_state": {int(B): states[B] for B in Bs},
               "log_cost_slope_per_doubling": slope, "slope_low": lo, "slope_high": hi,
               "flat": bool(lo <= 0 <= hi), "order": "interleaved, reversed every round",
               "seed": seed, "margin": MARGIN_H5, "below_margin": bool(hi < MARGIN_H5)}
        if all(len(norm[B]) == reps for B in Bs):
            cs, clo, chi = _slope_ci(norm, reps)
            row.update({"clock_normalised_slope": cs, "clock_normalised_low": clo, "clock_normalised_high": chi,
                        "clock_normalised_flat": bool(clo <= 0 <= chi)})
        sm = [g.get("clocks.sm") for B in Bs for g in states[B] if g.get("clocks.sm")]
        tc = [g.get("temperature.gpu") for B in Bs for g in states[B] if g.get("temperature.gpu")]
        clk = f" sm {min(sm):.0f}-{max(sm):.0f} MHz {min(tc):.0f}-{max(tc):.0f} C" if sm and tc else ""
        slope_rows.append(row)
        print(f"E6 {s.name:34s} rate@{Bs[-1]}={np.mean(data[Bs[-1]]):.3f}/ns slope={slope:+.3f} [{lo:+.3f},{hi:+.3f}]"
              + (f" clock-norm {row['clock_normalised_slope']:+.3f}" if "clock_normalised_slope" in row else "")
              + clk, flush=True)
    # paired comparison with ising-gpu: 2D square Ising at Tc, 4096 x 4096 spins both sides
    pair = {"status": "not run"}
    exe = build_ising_gpu(work, _sm())
    if exe is not None:
        s = sy.ising("square", 1024, T_ref=sy.TC["Ising square"])
        A = Sampler(s, 16, device, 1)
        A.set_temperature(sy.TC["Ising square"])
        S = A.init_spins(seed=1)
        k = 20
        replay = A.graph(S, k)
        replay()
        ours, base, pstate = [], [], []
        for i in range(reps):
            t0 = time.time()
            ms = _events_ms(replay, 1, A.device)
            pstate.append(mon.window(t0, time.time()))
            ours.append(16 * s.n_active * k / (ms * 1e6))
            base.append(run_ising_gpu(exe, 4096, 4096, 200))
        lr = np.log(np.asarray(ours) / np.asarray(base))
        m, lo, hi = boot_ci(lr)
        # the bit-packed Ising kernel needs a full word of replicas: 32 x 1024^2, same lattice per replica
        P32 = Sampler(s, 32, device, 1)
        P32.set_temperature(sy.TC["Ising square"])
        S32 = P32.init_spins(seed=1)
        rp = P32.graph(S32, k)
        rp()
        packed = [32 * s.n_active * k / (_events_ms(rp, 1, P32.device) * 1e6) for _ in range(reps)]
        pm, plo, phi = boot_ci(np.log(np.asarray(packed) / np.asarray(base)))
        pair = {"status": "run", "generated_flips_per_ns": ours, "ising_gpu_spins_per_ns": base,
                "log_ratio": m, "low": lo, "high": hi, "lattice": "16 x 1024^2 vs 4096^2", "gpu_state": pstate,
                "packed_flips_per_ns": packed, "packed_log_ratio": pm, "packed_low": plo, "packed_high": phi,
                "packed_lattice": "32 x 1024^2, multi-spin coded",
                "ising_gpu_commit": ISING_GPU_COMMIT}
        print(f"E6 pair: ours {np.mean(ours):.2f}/ns  ising-gpu {np.mean(base):.2f}/ns  log ratio {m:+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"  packed(B=32) {np.mean(packed):.2f}/ns log ratio {pm:+.3f} [{plo:+.3f},{phi:+.3f}]")
    mon.close()
    return save(out, tag, {"slopes": slope_rows, "pair": pair,
                            "params": {"batches": list(batches), "reps": reps}})


def e6_bitwise(out: Path, device="cuda", reps: int = 10, seed: int = 2006, work: Path = Path("/tmp/sck"),
               groups=(1, 4, 8, 16), tag: str = "E6bit", independent: bool = False):
    """Experiments v06, E6 (route B): the bitwise multi-spin Ising mode paired with NVIDIA ising-gpu.

    Same spins per sweep on both sides: 64 replicas of 512^2 (two 32-replica words) against one 4096^2
    lattice, 2D square Ising at Tc. A short pilot picks words_per_program among `groups` (fastest median of
    3 replays); then 10 repetitions alternate the two codes on the same device and the paired log speed ratio
    is bootstrapped. The pilot's rates are kept in the JSON."""
    mon = GpuMonitor(device)
    time.sleep(1.0)
    exe = build_ising_gpu(work, _sm())
    s = sy.ising("square", 512, T_ref=sy.TC["Ising square"])
    B, k = 64, 50
    pilot = {}
    for G in groups:
        try:
            A = Sampler(s, B, device, seed, bitwise=True, words_per_program=G, **({"independent": True} if independent else {}))
        except Exception as exc:              # a group size the launch shape cannot take
            pilot[G] = {"error": repr(exc)}
            continue
        A.set_temperature(sy.TC["Ising square"])
        S = A.init_spins(seed=seed)
        rp = A.graph(S, k)
        rp()
        pilot[G] = {"flips_per_ns": [B * s.n_active * k / (_events_ms(rp, 1, A.device) * 1e6) for _ in range(3)]}
        del A, S, rp
        torch.cuda.empty_cache()
    ok = {G: v for G, v in pilot.items() if "flips_per_ns" in v}
    best = max(ok, key=lambda G: float(np.median(ok[G]["flips_per_ns"])))
    A = Sampler(s, B, device, seed, bitwise=True, words_per_program=best, **({"independent": True} if independent else {}))
    A.set_temperature(sy.TC["Ising square"])
    S = A.init_spins(seed=seed)
    rp = A.graph(S, k)
    rp()
    ours, base, st = [], [], []
    for i in range(reps):
        t0 = time.time()
        ms = _events_ms(rp, 1, A.device)
        st.append(mon.window(t0, time.time()))
        ours.append(B * s.n_active * k / (ms * 1e6))
        if exe is not None:
            base.append(run_ising_gpu(exe, 4096, 4096, 200))
    res = {"bitwise_flips_per_ns": ours, "words_per_program": best, "pilot": pilot, "gpu_state": st, "independent": independent,
           "lattice": "64 x 512^2 (bitwise, 2 words) vs 4096^2", "seed": seed, "ising_gpu_commit": ISING_GPU_COMMIT}
    if base:
        m, lo, hi = boot_ci(np.log(np.asarray(ours) / np.asarray(base)))
        res.update({"ising_gpu_spins_per_ns": base, "log_ratio": m, "low": lo, "high": hi})
        print(f"E6bit: bitwise {np.mean(ours):.1f}/ns (G={best})  ising-gpu {np.mean(base):.1f}/ns  "
              f"log ratio {m:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    else:
        print(f"E6bit: bitwise {np.mean(ours):.1f}/ns (G={best}); ising-gpu not built", flush=True)
    mon.close()
    return save(out, tag, res)


def e6_independence(out: Path, device="cuda", L: int = 512, B: int = 64, therm: int = 2000, sweeps: int = 1000,
                    seed: int = 3008, tag: str = "E6indep"):
    """Experiments v08, E6: are the replicas of a bitwise word independent samples?

    2D square Ising at Tc, B replicas of L^2 in B / 32 words, shared mode and independent mode: after `therm`
    sweeps, the energy of every replica after each of `sweeps` sweeps; the Pearson correlation of every pair
    of replicas, split into pairs within a word and pairs across words. Criterion (independent mode): mean
    absolute within-word correlation below 0.1 and below the across-word mean plus 0.05."""
    s = sy.ising("square", L, T_ref=sy.TC["Ising square"])
    res = {}
    for mode in ("shared", "independent"):
        kw = {"independent": True} if mode == "independent" else {}
        A = Sampler(s, B, device, seed, bitwise=True, **kw)
        A.set_temperature(sy.TC["Ising square"])
        S = A.init_spins(seed=seed)
        A.sweep(S, therm)
        E = np.empty((sweeps, B))
        for t in range(sweeps):
            A.sweep(S, 1)
            E[t] = A.energy(A.unpack(S)).float().cpu().numpy().reshape(B)
        C = np.corrcoef(E.T)
        iu = np.triu_indices(B, 1)
        same = (iu[0] // 32) == (iu[1] // 32)
        within, across = np.abs(C[iu][same]), np.abs(C[iu][~same])
        distinct = [len({S_.tobytes() for S_ in A.unpack(S)[w * 32:(w + 1) * 32].cpu().numpy()}) for w in range(B // 32)]
        res[mode] = {"mean_abs_within": float(within.mean()), "mean_abs_across": float(across.mean()),
                     "max_abs_within": float(within.max()), "distinct_per_word_at_end": distinct}
        print(f"E6indep {mode:11s} within {within.mean():.3f} (max {within.max():.3f}) across {across.mean():.3f} "
              f"distinct per word {distinct}", flush=True)
        del A, S
        torch.cuda.empty_cache()
    ind = res["independent"]
    res["criterion_met"] = bool(ind["mean_abs_within"] < 0.1 and ind["mean_abs_within"] < ind["mean_abs_across"] + 0.05)
    res["params"] = {"L": L, "B": B, "therm": therm, "sweeps": sweeps, "seed": seed}
    return save(out, tag, res)


# ------------------------------------------------------------------ E7
def section0_launcher(geo_sys: sy.System, B: int, device, Rd=18.30, L=5, params=None, seed: int = 1):
    """Torch launcher for the vendored section0 kernel, on the same nanodot as `geo_sys`."""
    from .section0_kernel import _substep_kernel
    p = {"Jex": 1.0, "Kan2": 0.034, "Hex": 0.483, "KDM": 0.88, "g": 90.0}
    if params:
        p.update(params)
    N, LZ = geo_sys.L1, geo_sys.L3
    mask = torch.as_tensor(geo_sys.site_mask()[0], device=device)              # (LZ, N, N)
    disk = mask[0].to(torch.float32).contiguous()
    k1 = torch.as_tensor(geo_sys.cubic_k1[0], device=device).expand(B, LZ, N, N).contiguous()
    g = math.radians(p["g"])
    par = torch.tensor([p["Jex"], p["Kan2"], p["Hex"], p["KDM"], math.cos(g), math.sin(g)],
                       dtype=torch.float32, device=device).expand(B, 6).contiguous()
    step = torch.tensor([seed * 1000], dtype=torch.int32, device=device)
    dum = torch.zeros(1, dtype=torch.float32, device=device)
    BY, BX = 4, 1 << (N - 1).bit_length()
    grid = (B, LZ, -(-N // BY))

    def run(S, beta, sweeps):
        for _ in range(sweeps):
            for parity in (0, 1):
                _substep_kernel[grid](S, k1, par, disk, beta, step, dum, dum, S, dum,
                                      N=N, LZ=LZ, PARITY=parity, BY=BY, BX=BX, HAS_RNG=True,
                                      WRITE_DE=False, SO=1.0, num_warps=4)
                step.add_(1)
        return S
    return run


def e7(out: Path, device="cuda", B: int = 1024, reps: int = 10, T: float = 2.0, k: int = 10):
    """Generated kernel against the hand-specialized section0 kernel on the production nanodot (H6)."""
    s = sy.nanodot()
    dev = torch.device(device)
    A = Sampler(s, B, dev, 1)
    A.set_temperature(T)
    S_ours = A.init_spins(seed=1)
    S_s0 = S_ours[:, :, 0].clone().contiguous()            # (B, 3, LZ, N, N): section0 layout
    beta = torch.tensor([1.0 / T], dtype=torch.float32, device=dev)
    run0 = section0_launcher(s, B, dev)
    run0(S_s0, beta, 1)
    torch.cuda.synchronize(dev)
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        run0(S_s0, beta, k)
    ours_replay = A.graph(S_ours, k)
    ours_replay()
    g0.replay()
    upd = B * s.n_active * k
    ours, theirs = [], []
    for i in range(reps):
        if i % 2 == 0:
            ours.append(upd / (_events_ms(ours_replay, 1, dev) * 1e6))
            theirs.append(upd / (_events_ms(g0.replay, 1, dev) * 1e6))
        else:
            theirs.append(upd / (_events_ms(g0.replay, 1, dev) * 1e6))
            ours.append(upd / (_events_ms(ours_replay, 1, dev) * 1e6))
    lr = np.log(np.asarray(ours) / np.asarray(theirs))
    m, lo, hi = boot_ci(lr)
    # cross-check of the Hamiltonian: both chains equilibrated at T, mean energy per active site
    for _ in range(30):
        ours_replay()
        g0.replay()
    torch.cuda.synchronize(dev)
    e_ours = (A.energy(S_ours) / s.n_active).cpu().numpy()
    e_s0 = (A.energy(S_s0[:, :, None]) / s.n_active).cpu().numpy()
    res = {"generated_flips_per_ns": ours, "section0_flips_per_ns": theirs, "log_ratio": m, "low": lo,
           "high": hi, "batch": B, "active_sites": s.n_active, "T": T,
           "energy_per_site": {"generated": boot_ci(e_ours), "section0": boot_ci(e_s0)},
           "colours": s.colouring.n_colours}
    print(f"E7 ours {np.mean(ours):.3f}/ns section0 {np.mean(theirs):.3f}/ns log ratio {m:+.3f} [{lo:+.3f},{hi:+.3f}]; "
          f"E/site {e_ours.mean():.4f} vs {e_s0.mean():.4f}", flush=True)
    return save(out, "E7", res)


# ------------------------------------------------------------------ E8 (supplementary, two GPUs)
def e8(out: Path, reps: int = 10, work: Path = Path("/tmp/sck")):
    """Replica batch split over every visible GPU against one GPU; ising-gpu with -g 2 as the pair."""
    ng = torch.cuda.device_count()
    rows = []
    cases = [(sy.ising("square", 1024, T_ref=sy.TC["Ising square"]), 16),
             (sy.heisenberg("sc", 32, T_ref=sy.TC["Heisenberg sc"]), 128),
             (sy.nanodot(), 1024)]
    for s, B in cases:
        res = {}
        for g in sorted({1, ng}):
            per = B // g
            samplers, replays = [], []
            k = max(1, int(math.ceil(2e7 / (per * s.n_active))))
            for d in range(g):
                A = Sampler(s, per, torch.device(f"cuda:{d}"), 1 + d)
                A.set_temperature(s.T_ref or 1.0)
                S = A.init_spins(seed=1 + d)
                replays.append(A.graph(S, k))
                samplers.append((A, S))
            for r in replays:
                r()
            vals = []
            for _ in range(reps):
                for d in range(g):
                    torch.cuda.synchronize(d)
                t0 = time.perf_counter()
                for r in replays:
                    r()
                for d in range(g):
                    torch.cuda.synchronize(d)
                vals.append(B * s.n_active * k / ((time.perf_counter() - t0) * 1e9))
            res[g] = vals
        row = {"system": s.name, "batch_total": B, "flips_per_ns": res}
        if ng > 1:
            row["scaling"] = boot_ci(np.asarray(res[ng]) / np.mean(res[1]))
        rows.append(row)
        print(f"E8 {s.name:30s} " + "  ".join(f"{g} GPU {np.mean(v):.2f}/ns" for g, v in res.items()), flush=True)
    pair = {"status": "not run"}
    exe = build_ising_gpu(work, _sm())
    if exe is not None and ng > 1:
        pair = {"status": "run", "ising_gpu_1gpu": [run_ising_gpu(exe, 4096, 4096, 200, 1) for _ in range(reps)],
                "ising_gpu_2gpu": [run_ising_gpu(exe, 4096, 2048, 200, 2) for _ in range(reps)]}
    return save(out, "E8", {"rows": rows, "gpus": ng, "ising_gpu": pair,
                            "note": "supplementary to experiments v03 (multi-GPU scaling is listed as left out)"})


# ------------------------------------------------------------------ E4
def e4_systems():
    return [
        (sy.ising, ("square",), {"T_ref": sy.TC["Ising square"]}, "2d", 3.0),        # 1/nu + omega
        (sy.ising, ("triangular",), {"T_ref": sy.TC["Ising triangular"]}, "2d", 3.0),
        (sy.ising, ("sc",), {"T_ref": sy.TC["Ising sc"]}, "3d", 2.42),
        (sy.ising, ("bcc",), {"T_ref": sy.TC["Ising bcc"]}, "3d", 2.42),
        (sy.ising, ("fcc",), {"T_ref": sy.TC["Ising fcc"]}, "3d", 2.42),
        (sy.xy, ("sc",), {"T_ref": sy.TC["XY sc"]}, "3d", 2.28),
        (sy.heisenberg, ("sc",), {"T_ref": sy.TC["Heisenberg sc"]}, "3d", 2.19),
        (sy.heisenberg, ("bcc",), {"T_ref": sy.TC["Heisenberg bcc"]}, "3d", 2.19),
    ]


def _series(s: sy.System, L_box, temps, seeds, therm, meas, every, device):
    """Run all temperatures and seeds of one size in one batch; returns E, m2, m4 series (T, seed, t)."""
    B = len(temps) * seeds
    A = Sampler(s, B, device, 12345 + L_box)
    T = torch.tensor(np.repeat(temps, seeds), dtype=torch.float32, device=device)
    A.set_temperature(T)
    S = A.init_spins(seed=L_box)
    N = s.n_active
    k = max(1, every)
    # a CUDA graph of k sweeps is the same chain as k calls of `sweep` (the CPU/interpreter path)
    rep = A.graph(S, k) if str(device).startswith("cuda") else (lambda: A.sweep(S, k))
    for _ in range(int(math.ceil(therm / k))):
        rep()
    E, m2, m4 = [], [], []
    for _ in range(int(math.ceil(meas / k))):
        rep()
        M = A.magnetization(S).sum(dim=1)        # (B, n), total magnetization
        m = (M.norm(dim=1) / N)
        E.append(A.energy(S).cpu().numpy())
        m2.append((m ** 2).cpu().numpy())
        m4.append((m ** 4).cpu().numpy())
    sh = (len(temps), seeds, -1)
    f = lambda x: np.stack(x, axis=1).reshape(sh)
    return f(E), f(m2), f(m4)


def multi_histogram(E, obs, temps, grid, weights=None, device="cpu", lnZ0=None, tol=1e-10, iters=5000,
                    return_lnZ=False):
    """Multi-histogram (Ferrenberg-Swendsen / WHAM) reweighting of all temperatures at once.

    E: energies (temps, samples); obs: list of observables of the same shape; temps: simulated
    temperatures; grid: temperatures to evaluate at. `weights` are per-sample multiplicities (bootstrap
    resampling). Returns one array per observable with its canonical average on `grid`.
    """
    dt = torch.float64
    E_ = torch.as_tensor(np.asarray(E, dtype=np.float64), dtype=dt, device=device)
    nT, nS = E_.shape
    w = (torch.ones_like(E_) if weights is None else
         torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=dt, device=device))
    b = torch.as_tensor(1.0 / np.asarray(temps, dtype=np.float64), dtype=dt, device=device)
    e0 = float((E_ * w).sum() / w.sum())
    x = (E_ - e0).reshape(-1)                         # shifted energies: exponents stay small
    lw = torch.log(w.reshape(-1))                     # -inf for dropped samples
    lnN = torch.log(w.sum(dim=1))                     # effective count per temperature
    if lnZ0 is None:                                  # start from thermodynamic integration
        Em = (E_ * w).sum(dim=1) / w.sum(dim=1) - e0
        lnZ = torch.zeros(nT, dtype=dt, device=device)
        lnZ[1:] = torch.cumsum(-(b[1:] - b[:-1]) * 0.5 * (Em[1:] + Em[:-1]), 0)
    else:
        lnZ = torch.as_tensor(lnZ0, dtype=dt, device=device).clone()
    bx = b[:, None] * x[None, :]                      # (temps, all samples)
    for _ in range(iters):
        D = torch.logsumexp(lnN[:, None] - bx - lnZ[:, None], dim=0)
        new = torch.logsumexp(lw[None, :] - bx - D[None, :], dim=1)
        new = new - new[0]
        done = float((new - lnZ).abs().max()) < tol
        lnZ = new
        if done:
            break
    D = torch.logsumexp(lnN[:, None] - bx - lnZ[:, None], dim=0)
    g = torch.as_tensor(1.0 / np.asarray(grid, dtype=np.float64), dtype=dt, device=device)
    lp = lw[None, :] - g[:, None] * x[None, :] - D[None, :]      # (grid, samples)
    p = torch.softmax(lp, dim=1)
    out = [(p @ torch.as_tensor(np.asarray(o, dtype=np.float64), dtype=dt, device=device).reshape(-1)).cpu().numpy()
           for o in obs]
    return (out, lnZ.cpu().numpy()) if return_lnZ else out


def e4_temperatures(Tc: float, sigma_E: float, ntemps: int, window: float):
    """Simulated temperatures for one size, centred on Tc.

    Neighbours sit one canonical energy width apart (delta beta * sigma_E = 1, i.e. delta T / T = T / sigma_E)
    so the multi-histogram reweighting has overlapping histograms; the span never exceeds +-window.
    """
    half = min(window, 0.5 * (ntemps - 1) * Tc / sigma_E)
    return Tc * (1 + np.linspace(-half, half, ntemps))


def _binder_curve(E, m2, m4, temps, grid, weights=None, device="cpu", lnZ0=None):
    """Binder cumulant on `grid` from all temperatures by multi-histogram reweighting."""
    (a2, a4), lnZ = multi_histogram(E.reshape(len(temps), -1), [m2.reshape(len(temps), -1),
                                                                m4.reshape(len(temps), -1)],
                                    temps, grid, device=device, lnZ0=lnZ0, return_lnZ=True,
                                    weights=None if weights is None else np.reshape(weights, (len(temps), -1)))
    return 1.0 - a4 / (3.0 * a2 * a2), lnZ


def _crossing(grid, u1, u2):
    d = u1 - u2
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(idx) == 0:
        return np.nan
    j = idx[len(idx) // 2]
    return float(grid[j] - d[j] * (grid[j + 1] - grid[j]) / (d[j + 1] - d[j]))


def _probe_tau(s, Lb, Tc, seeds, device, start=2000, cap=100_000, ratio=50):
    """Integrated autocorrelation time of m^2 at Tc from a probe that grows until it is >= ratio * tau.

    Returns tau, the probe length used, whether that length reached ratio * tau, and the energy spread
    of one configuration (sigma_E, used to place the temperatures)."""
    P = start
    while True:
        E, m2, _ = _series(s, Lb, [Tc], seeds, max(500, P // 4), P, 1, device)
        tau = integrated_tau(m2[0].T)
        if P >= ratio * tau or P >= cap:
            return tau, P, bool(P >= ratio * tau), float(E[0].std(axis=1).mean())
        P = min(cap, 4 * P)


def e4(out: Path, device="cuda", sizes2d=(32, 64, 128, 256), sizes3d=(16, 24, 32, 48), seeds: int = 10,
       window: float = 0.03, ntemps: int = 9, meas: int = 20000, boots: int = 2000, only=None, tag="E4",
       probe_cap: int = 100_000):
    """Binder-cumulant crossings with bootstrap over seeds and size extrapolation (H3).

    Per size: tau from an adaptive probe at Tc; ntemps temperatures one energy width apart (capped at
    +-window); multi-histogram reweighting over all of them; raw series saved to <tag>-<system>-L<L>.npz.
    """
    from .exp_core import holm
    rows = []
    adev = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    for maker, args, kw, dim, x in e4_systems():
        probe = maker(*args, 6 if args[0] in ("triangular",) else 4, **kw)
        if only and probe.name not in only:
            continue
        Tc = kw["T_ref"]
        sizes = sizes2d if dim == "2d" else sizes3d
        data = {}
        t_start = time.time()
        for L in sizes:
            Lb = L if probe.crystal.name != "triangular" else 3 * int(round(L / 3))
            s = maker(*args, Lb, **kw)
            tau, P, resolved, sigma_E = _probe_tau(s, Lb, Tc, seeds, device, cap=probe_cap)
            temps = e4_temperatures(Tc, sigma_E, ntemps, window)
            therm = int(max(1000, 20 * tau))
            every = max(1, int(round(tau / 2)), meas // 2000)   # at most 2000 measurements per chain
            E, m2, m4 = _series(s, Lb, temps, seeds, therm, meas, every, device)
            mid = int(np.argmin(np.abs(temps - Tc)))
            tau_prod = every * integrated_tau(m2[mid].T)          # in sweeps, from the production series
            np.savez_compressed(out / f"{tag}-{probe.name.replace(' ', '_')}-L{Lb}.npz",
                                E=E, m2=m2.astype(np.float32), m4=m4.astype(np.float32), temps=temps,
                                tau=tau, therm=therm, every=every)
            data[Lb] = {"E": E, "m2": m2, "m4": m4, "temps": temps, "tau": tau, "probe_sweeps": P,
                        "tau_resolved": resolved, "sigma_E": sigma_E, "therm": therm, "every": every,
                        "tau_production": tau_prod, "therm_over_tau_production": therm / tau_prod}
            print(f"E4 {probe.name:20s} L={Lb} tau={tau:.1f} (probe {P}, resolved {resolved}) "
                  f"tau_prod={tau_prod:.1f} therm={therm} every={every} "
                  f"window=+-{(temps[-1] / Tc - 1):.4%} {time.time() - t_start:.0f}s", flush=True)
        Ls = sorted(data)
        pairs_LL = list(zip(Ls[:-1], Ls[1:]))
        grids = {}
        for a_, b_ in pairs_LL:      # each pair is compared only where both sizes have reweightable data
            lo_ = max(data[a_]["temps"][0], data[b_]["temps"][0])
            hi_ = min(data[a_]["temps"][-1], data[b_]["temps"][-1])
            grids[(a_, b_)] = np.linspace(lo_, hi_, 241)
        lnZ_full = {}

        def curves_for(sel_w):
            cur = {}
            for (a_, b_), g in grids.items():
                for L_ in (a_, b_):
                    d = data[L_]
                    w = None if sel_w is None else np.broadcast_to(sel_w[None, :, None], d["E"].shape)
                    u, lnZ = _binder_curve(d["E"], d["m2"], d["m4"], d["temps"], g, weights=w, device=adev,
                                           lnZ0=lnZ_full.get(L_))
                    if sel_w is None:
                        lnZ_full[L_] = lnZ
                    cur[(a_, b_, L_)] = u
            return cur

        def estimate(sel_w):
            cur = curves_for(sel_w)
            cr_ = [(math.sqrt(a_ * b_), _crossing(grids[(a_, b_)], cur[(a_, b_, a_)], cur[(a_, b_, b_)]))
                   for a_, b_ in pairs_LL]
            cr_ = [(l, t_) for l, t_ in cr_ if np.isfinite(t_)]
            if len(cr_) >= 2:
                X = np.array([l ** (-x) for l, _ in cr_])
                Y = np.array([t_ for _, t_ in cr_])
                return float(np.polyfit(X, Y, 1)[1]), cr_, cur
            return (cr_[-1][1] if cr_ else np.nan), cr_, cur

        est, pairs, cur_full = estimate(None)
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(boots):
            w = np.bincount(rng.integers(0, seeds, seeds), minlength=seeds).astype(np.float64)
            bs.append(estimate(w)[0])
        bs = np.array(bs)
        bs = bs[np.isfinite(bs)]
        lo, hi = (np.quantile(bs, [0.025, 0.975]) if len(bs) > 10 else (np.nan, np.nan))
        half = (hi - lo) / 2 / Tc
        # two-sided bootstrap p-value of the reference against the bootstrap distribution
        p = float(min(1.0, 2 * min(np.mean(bs <= Tc), np.mean(bs >= Tc)))) if len(bs) else np.nan
        keys = ("tau", "probe_sweeps", "tau_resolved", "sigma_E", "therm", "every", "tau_production",
                "therm_over_tau_production")
        rows.append({"system": probe.name, "T_ref": Tc, "estimate": est, "low": float(lo), "high": float(hi),
                     "half_width_over_Tc": float(half), "p_reference": p, "pair_crossings": pairs,
                     "sizes": Ls, "temps": {L: data[L]["temps"].tolist() for L in Ls},
                     "correction_exponent": x, "reweighting": "multi-histogram",
                     "declared": {L: {k: data[L][k] for k in keys} for L in Ls},
                     "binder": {f"{a_}-{b_}": {"grid": grids[(a_, b_)].tolist(),
                                              str(a_): cur_full[(a_, b_, a_)].tolist(),
                                              str(b_): cur_full[(a_, b_, b_)].tolist()} for a_, b_ in pairs_LL},
                     "seconds": time.time() - t_start})
        print(f"E4 {probe.name:20s} Tc={est:.5f} [{lo:.5f},{hi:.5f}] ref {Tc:.5f} half {half:.4%}", flush=True)
    padj = holm([r["p_reference"] if np.isfinite(r["p_reference"]) else 1.0 for r in rows]) if rows else []
    for r, pa in zip(rows, padj):
        r["p_holm"] = pa
        r["verdict"] = ("inconclusive" if not (r["half_width_over_Tc"] <= 0.005) else
                        ("supported" if pa >= 0.05 else "not supported"))
    return save(out, tag, {"rows": rows, "params": {"sizes2d": list(sizes2d), "sizes3d": list(sizes3d),
                                                   "seeds": seeds, "window": window, "ntemps": ntemps,
                                                   "meas": meas, "boots": boots, "probe_cap": probe_cap,
                                                   "reweighting": "multi-histogram"}})


def compaction_ab(out: Path, device="cuda", rounds: int = 10, reps: int = 5, tag="E-compact"):
    """Same-machine A/B of colour compaction (on against off), interleaved, on the boxes it was built for.

    The T4 A/B (2026-11-07) ran under the 70 W power cap and saw Heisenberg on 62 x 31^2 regress; this
    repeats it where the clock holds. The statistic is log(rate on / rate off) per round, bootstrapped.
    """
    from . import crystals as cr
    c = cr.make("sc")
    cases = []
    for n, name in ((1, "Ising"), (3, "Heisenberg")):
        for box in ((62, 31, 31), (31, 31, 31), (64, 32, 32)):
            for B in ((21, 128) if n == 3 else (32, 128)):
                cases.append((f"{name} sc {box[0]}x{box[1]}x{box[2]}", n, box, B))
    rows = []
    for label, n, box, B in cases:
        s = sy.System(label, n, c, box, (True,) * 3, {1: 1.0}, T_ref=4.5 if n == 1 else 1.44)
        on, off = [], []
        for r in range(rounds):
            order = (True, False) if r % 2 == 0 else (False, True)
            for cp in order:
                x = float(np.median(rate(s, B, device, reps=reps, compact=cp)[0]))
                (on if cp else off).append(x)
        lr = np.log(np.array(on)) - np.log(np.array(off))
        st, lo, hi = boot_ci(lr)
        A = Sampler(s, B, device, 1)
        rows.append({"system": label, "box": list(box), "B": B, "stride": A.stride, "bx": A.bx,
                     "rows_per": A.rows_per, "on": on, "off": off, "log_ratio_on_off": st, "low": lo,
                     "high": hi, "gpu": gpu_state(device)})
        print(f"E-compact {label:28s} B={B:4d} P={A.stride} BX={A.bx} R={A.rows_per} "
              f"on {np.median(on):.3f} off {np.median(off):.3f} /ns  log on/off {st:+.3f} [{lo:+.3f},{hi:+.3f}]",
              flush=True)
    return save(out, tag, {"rows": rows, "params": {"rounds": rounds, "reps": reps}})


def packing_ab(out: Path, device="cuda", rounds: int = 10, reps: int = 5, sizes=(256, 1024),
               batches=(32, 128), tag="E-packed"):
    """Same-machine A/B of Ising multi-spin coding (packed against int8) at equal batch, interleaved.

    On the T4 (job v) packed B = 32 on 2D 1024^2 ran at half the int8 rate; the H100 ising-gpu pair
    compared packed B = 32 with int8 B = 16 only. Statistic: log(rate packed / rate int8) per round.
    """
    rows = []
    for L in sizes:
        s = sy.ising("square", L)
        for B in batches:
            pk, i8 = [], []
            for r in range(rounds):
                order = (True, False) if r % 2 == 0 else (False, True)
                for p_ in order:
                    x = float(np.median(rate(s, B, device, T=2.269, reps=reps, packed=p_)[0]))
                    (pk if p_ else i8).append(x)
            lr = np.log(np.array(pk)) - np.log(np.array(i8))
            st, lo, hi = boot_ci(lr)
            rows.append({"system": f"Ising square {L}^2", "L": L, "B": B, "packed": pk, "int8": i8,
                         "log_ratio_packed_int8": st, "low": lo, "high": hi, "gpu": gpu_state(device)})
            print(f"E-packed L={L:5d} B={B:4d} packed {np.median(pk):.2f} int8 {np.median(i8):.2f} /ns "
                  f"log {st:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    return save(out, tag, {"rows": rows, "params": {"rounds": rounds, "reps": reps}})


def _sk_series(s: sy.System, L: int, temps, seeds, therm, meas, every, device, seed):
    """K-point structure factor and energy series of the triangular AFM, shape (T, seed, t) each."""
    B = len(temps) * seeds
    A = Sampler(s, B, device, seed)
    A.set_temperature(torch.tensor(np.repeat(temps, seeds), dtype=torch.float32, device=device))
    S = A.init_spins(seed=L)
    pos = np.array([[s.crystal.position((x, y), 0) for x in range(L)] for y in range(L)])   # (L2, L1, 2)
    K = np.array([4 * np.pi / 3, 0.0])
    ph = torch.as_tensor(np.exp(1j * (pos @ K)), device=S.device)                          # (L2, L1)
    k = max(1, every)
    rep = A.graph(S, k) if str(device).startswith("cuda") else (lambda: A.sweep(S, k))
    for _ in range(int(math.ceil(therm / k))):
        rep()
    sk, E = [], []
    for _ in range(int(math.ceil(meas / k))):
        rep()
        F = (S[:, :, 0, 0].to(torch.complex128) * ph).sum(dim=(2, 3))                     # (B, n)
        sk.append(((F.abs() ** 2).sum(dim=1) / (L * L)).cpu().numpy())
        E.append(A.energy(S).cpu().numpy())
    sh = (len(temps), seeds, -1)
    return np.stack(sk, axis=1).reshape(sh), np.stack(E, axis=1).reshape(sh)


def sk_growth(means: dict, temps, crossover: float, boots: int = 2000, seed: int = 1):
    """Bootstrap test of the K-point peak against size.

    means maps L to an array (T, seeds) of per-chain time averages of S(K). For each temperature and
    adjacent size pair the statistic is log(mean S_K(L2) / mean S_K(L1)), resampling the seeds of each
    size independently (the chains of different sizes are independent). Below the crossover every pair
    must grow (lower bound > 0); above it the largest pair must not (lower bound <= 0), because a finite
    correlation length may still exceed the smaller sizes.
    """
    rng = np.random.default_rng(seed)
    Ls = sorted(means)
    temps = list(temps)
    tests = []
    for i, T in enumerate(temps):
        for a, b in zip(Ls[:-1], Ls[1:]):
            xa, xb = np.asarray(means[a][i], float), np.asarray(means[b][i], float)
            est = float(np.log(xb.mean() / xa.mean()))
            bs = np.array([np.log(xb[rng.integers(0, len(xb), len(xb))].mean() /
                                  xa[rng.integers(0, len(xa), len(xa))].mean()) for _ in range(boots)])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            tests.append({"T": float(T), "L1": int(a), "L2": int(b), "log_ratio": est, "low": float(lo),
                          "high": float(hi), "grows": bool(lo > 0), "below_crossover": bool(T < crossover)})
    below_ok = all(t["grows"] for t in tests if t["below_crossover"])
    above_ok = all(not t["grows"] for t in tests if not t["below_crossover"] and t["L2"] == Ls[-1])
    verdict = "supported" if below_ok and above_ok else "not supported"
    return {"tests": tests, "below_all_grow": below_ok, "above_largest_pair_flat": above_ok, "verdict": verdict}


def e4_triangular_afm(out: Path, device="cuda", sizes=(24, 48, 96), temps=(0.2, 0.25, 0.32, 0.4), seeds=10,
                      meas=100_000, boots=2000, crossover=0.285, probe_cap=100_000, tag="E4-tri",
                      seed_base: int = 0, thin: str = "slowest"):
    """K-point structure factor of the triangular Heisenberg antiferromagnet against size.

    Per size: tau of S(K) from an adaptive probe over all temperatures (the slowest one counts), then
    therm = max(1000, 20 tau); the per-chain series are saved to <tag>-L<L>.npz and the growth of the
    peak is tested with sk_growth.

    thin = "slowest" spaces the stored samples by the slowest temperature's tau (v03-v06); "per_T"
    (experiments v07) stores them at the fastest temperature's interval, at most 20 000 per chain, so every
    temperature keeps a well-sampled series (slow temperatures get correlated samples, which leaves the
    per-chain mean unbiased; the test bootstraps over independent seeds).
    """
    rows, means = [], {}
    temps = np.asarray(temps, dtype=float)
    t0 = time.time()
    for L in sizes:
        s = sy.heisenberg("triangular", L, J=-1.0, label="Heisenberg triangular AFM")
        P = 2000
        while True:
            sk, _ = _sk_series(s, L, temps, seeds, max(500, P // 4), P, 10, device, seed_base + 7 + L)
            taus = [10 * integrated_tau(sk[i].T) for i in range(len(temps))]
            tau = max(taus)
            if P >= 50 * tau or P >= probe_cap:
                break
            P = min(probe_cap, 4 * P)
        resolved = bool(P >= 50 * tau)
        therm = int(max(1000, 20 * tau))
        if thin == "per_T":
            every = max(10, int(round(min(taus) / 2)), meas // 20000)
        else:
            every = max(10, int(round(tau / 2)), meas // 2000)
        sk, E = _sk_series(s, L, temps, seeds, therm, meas, every, device, seed_base + 99 + L)
        tau_prod = [every * integrated_tau(sk[i].T) for i in range(len(temps))]
        np.savez_compressed(out / f"{tag}-L{L}.npz", S_K=sk.astype(np.float32), E=E, temps=temps,
                            therm=therm, every=every, tau=tau, thin=thin)
        m = sk.mean(axis=2)                                     # (T, seeds)
        means[L] = m
        rows.append({"L": L, "temps": temps.tolist(), "S_K": [boot_ci(m[i]) for i in range(len(temps))],
                     "tau_probe_per_T": taus, "tau": tau, "probe_sweeps": P, "tau_resolved": resolved,
                     "therm": therm, "every": every, "meas": meas, "tau_production_per_T": tau_prod,
                     "therm_over_tau_production": therm / max(tau_prod), "thin": thin,
                     "samples_per_chain": int(sk.shape[2])})
        print(f"E4-tri L={L} tau={tau:.0f} (probe {P}, resolved {resolved}) therm={therm} every={every} "
              + " ".join(f"T={t}:{m[i].mean():.1f}" for i, t in enumerate(temps))
              + f" {time.time() - t0:.0f}s", flush=True)
    g = sk_growth(means, temps, crossover, boots)
    for t in g["tests"]:
        print(f"E4-tri T={t['T']} {t['L1']}->{t['L2']} log ratio {t['log_ratio']:+.3f} "
              f"[{t['low']:+.3f},{t['high']:+.3f}]", flush=True)
    print(f"E4-tri verdict {g['verdict']}", flush=True)
    # finite-size slope of log S(K) against log L per temperature (2 - eta = 1.75 marks quasi-long-range order)
    Ls = np.array(sorted(means), float)
    slopes = []
    for i, t in enumerate(temps):
        y = np.log([means[int(L)][i].mean() for L in Ls])
        slopes.append({"T": float(t), "slope": float(np.polyfit(np.log(Ls), y, 1)[0])})
    g["fss_slopes"] = slopes
    print("E4-tri FSS slopes " + " ".join(f"T={x['T']}:{x['slope']:.2f}" for x in slopes), flush=True)
    return save(out, tag, {"rows": rows, "crossover": crossover, "growth": g,
                           "params": {"sizes": list(sizes), "seeds": seeds, "meas": meas, "boots": boots,
                                      "probe_cap": probe_cap, "thin": thin, "seed_base": seed_base}})
