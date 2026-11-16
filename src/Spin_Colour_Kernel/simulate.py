"""General simulation of a user-defined spin Hamiltonian, behind notebooks/spin_simulation.ipynb.

A run is one model (Ising, XY or Heisenberg), one crystal, one box, one set of active sites and a list
of states; a state is one parameter set theta of the Hamiltonian

    H = - sum_<ij> J_s  S_i . S_j                       exchange, per neighbour shell s
        - sum_<ij> S_i^T A_ij S_j                       bulk-type (Bloch) DMI along the bond, A_lm = D sum_k r_k eps_klm
                                                        (in-plane first-shell bonds, Heisenberg only)
        + sum_i K1_i (sx^2 sy^2 + sx^2 sz^2 + sy^2 sz^2) + K2 sum_i (sx sy sz)^2     cubic anisotropy
        - sum_i H . S_i                                 Zeeman field

with K1_i = Ks on surface sites (fewer active first-shell neighbours than the bulk). Every replica of a
state follows the same temperature ladder, from T_max down to T_min in steps of T_step; at each
temperature it thermalizes, then measures every `measure_every` sweeps.

theta keys: J {shell: value}, D, K1, Ks, K2, H, H_theta, H_phi. The field direction is
(sin t cos p, sin t sin p, cos t) for Heisenberg (t = H_theta, p = H_phi, degrees; t = 0 is +z),
(cos p, sin p) for XY and +1 for Ising.

Outputs, in the run directory:
    run.json             the configuration and the derived ladder, sites and units
    sites.npy            (n_sites, 3) Cartesian positions of the active sites
    state_XXX_pYY.npz    per state and replica block: r0, r1, M (B, nT, 3), E, E2, Mabs, M2 (B, nT)
    configurations.npy   (n_states, B, n_saved, n_sites, 3) float32 spins every config_every T steps
    magnetization.npy    (n_states, B, nT, 3), written by `collect`
    results.npz          every combined array, Cv and chi included, written by `collect`

Run as a module to process a block of replicas of some states of a prepared run on one device (the
notebook gives every GPU an equal block of the replicas of every state):
    python -m Spin_Colour_Kernel.simulate --run DIR --states 0,1 --replicas 0:16 --part 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import crystals as cr
from .sampler import Sampler
from .system import System

KB_MEV = 0.086173404          # meV / K
MODELS = {"ising": 1, "xy": 2, "heisenberg": 3}
COORDS = {"cartesian": ("x", "y", "z"), "spherical": ("r", "theta", "phi"),
          "cylindrical": ("rho", "phi", "z")}
_MATH = {k: getattr(np, k) for k in ("sqrt", "sin", "cos", "tan", "arctan2", "arccos", "abs", "exp",
                                     "log", "minimum", "maximum", "pi", "mod", "floor", "ceil")}
_MATH["abs"] = np.abs


# ------------------------------------------------------------------ geometry
def _positions(crystal: cr.Crystal, box) -> np.ndarray:
    """Cartesian position of every site, (b, L3, L2, L1, 3), centred on the box."""
    D = crystal.D
    L1, L2 = int(box[0]), int(box[1])
    L3 = int(box[2]) if D == 3 else 1
    z, y, x = np.meshgrid(np.arange(L3), np.arange(L2), np.arange(L1), indexing="ij")
    cells = np.stack([x, y, z][:D], axis=-1).astype(float)          # (L3, L2, L1, D)
    out = np.zeros((crystal.b, L3, L2, L1, 3))
    for a in range(crystal.b):
        out[a, ..., :D] = (cells + crystal.basis[a]) @ crystal.vectors
    return out - out.reshape(-1, 3).mean(axis=0)


def active_sites(crystal, box, coords: str = "", shape: str = ""):
    """(positions (b, L3, L2, L1, 3), mask (b, L3, L2, L1)) for a shape written in a coordinate system.

    coords is "cartesian" (x, y, z), "spherical" (r, theta, phi) or "cylindrical" (rho, phi, z), with the
    origin at the centre of the box and angles in radians; shape is a condition in those names, for
    example "r <= 10" or "(rho <= 8) & (abs(z) <= 2)". Both empty: every site is active.
    """
    c = crystal if isinstance(crystal, cr.Crystal) else cr.make(str(crystal))
    if len(box) != c.D:
        raise ValueError(f"crystal {c.name} needs a box of {c.D} sizes, got {list(box)}")
    pos = _positions(c, box)
    coords, shape = (coords or "").strip().lower(), (shape or "").strip()
    if not shape:
        if coords and coords not in COORDS:
            raise ValueError(f"unknown coordinate system {coords!r}; known: {', '.join(COORDS)}")
        return pos, np.ones(pos.shape[:-1], bool)
    if not coords:
        raise ValueError("a shape needs its coordinate system: set coords to one of " + ", ".join(COORDS))
    if coords not in COORDS:
        raise ValueError(f"unknown coordinate system {coords!r}; known: {', '.join(COORDS)}")
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    rho = np.hypot(x, y)
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan2(y, x)
    names = {"cartesian": dict(x=x, y=y, z=z),
             "spherical": dict(r=r, theta=np.arccos(np.divide(z, r, out=np.ones_like(z), where=r > 0)), phi=phi),
             "cylindrical": dict(rho=rho, phi=phi, z=z)}[coords]
    try:
        mask = eval(shape, {"__builtins__": {}}, {**_MATH, **names})   # the user's own notebook cell
    except NameError as e:
        raise ValueError(f"{e}; {coords} coordinates are {', '.join(COORDS[coords])}") from None
    mask = np.broadcast_to(np.asarray(mask, bool), pos.shape[:-1]).copy()
    if not mask.any():
        raise ValueError(f"the shape {shape!r} contains no site of the box")
    return pos, mask


# ------------------------------------------------------------------ Hamiltonian
def _f(theta, key, default=0.0) -> float:
    v = theta.get(key, default)
    return float(default if v is None else v)


def _dmi(crystal: cr.Crystal, D: float) -> dict[int, np.ndarray]:
    """Bulk-type (Bloch) DMI: A_lm = D sum_k r_k eps_klm on in-plane first-shell bonds (r the unit bond vector)."""
    eps = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, k], eps[i, k, j] = 1.0, -1.0
    out = {}
    for i, bd in enumerate(crystal.bonds):
        if bd.shell != 1:
            continue
        r = np.zeros(3)
        r[:crystal.D] = (np.asarray(bd.delta, float) + crystal.basis[bd.alpha2] - crystal.basis[bd.alpha]) @ crystal.vectors
        if abs(r[2]) > 1e-9:
            continue
        r /= np.linalg.norm(r)
        out[i] = D * np.einsum("k,klm->lm", r, eps)
    return out


def _surface(crystal: cr.Crystal, box, periodic, mask: np.ndarray) -> np.ndarray:
    """Active sites with fewer active first-shell neighbours than a bulk site of their basis index."""
    full = {a: 0 for a in range(crystal.b)}
    count = np.zeros(mask.shape, int)
    axes = {0: 3, 1: 2, 2: 1}                      # lattice axis -> array axis of (b, L3, L2, L1)
    for bd in crystal.bonds:
        if bd.shell != 1:
            continue
        full[bd.alpha] += 1
        src = mask[bd.alpha2].astype(int)
        valid = np.ones(mask.shape[1:], int)
        for k in range(crystal.D):
            d = bd.delta[k]
            if d == 0:
                continue
            src = np.roll(src, -d, axis=axes[k] - 1)
            if not periodic[k]:
                idx = np.arange(box[k])
                ok = ((idx + d) >= 0) & ((idx + d) < box[k])
                shp = [1, 1, 1]
                shp[2 - k] = box[k]
                valid = valid * ok.astype(int).reshape(shp)
        count[bd.alpha] += src * valid
    bulk = np.stack([np.full(mask.shape[1:], full[a]) for a in range(crystal.b)])
    return mask & (count < bulk)


def build_system(model: str, crystal: str, box, periodic, theta: dict, mask=None, log=None) -> System:
    """The System of one state theta. A periodic box with no admissible colouring falls back to open."""
    model = str(model).strip().lower()
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {', '.join(MODELS)}")
    n = MODELS[model]
    J = {int(k): float(v) for k, v in (theta.get("J") or {1: 1.0}).items()}
    D, K1, K2, H = _f(theta, "D"), _f(theta, "K1"), _f(theta, "K2"), _f(theta, "H")
    Ks = _f(theta, "Ks", K1)
    if n != 3 and any(_f(theta, k) for k in ("D", "K1", "Ks", "K2")):
        raise ValueError(f"D, K1, Ks and K2 need the Heisenberg model; {model} carries J and H only")
    unknown = set(theta) - {"name", "J", "D", "K1", "Ks", "K2", "H", "H_theta", "H_phi"}
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; known: J, D, K1, Ks, K2, H, H_theta, H_phi")
    shells = tuple(sorted({s for s, v in J.items() if v != 0.0} | {1}))
    c = cr.make(str(crystal), shells)
    box = tuple(int(v) for v in box)
    periodic = tuple(bool(v) for v in periodic)
    m = None if mask is None else np.asarray(mask, bool)
    if m is not None and m.shape != (c.b, box[2] if c.D == 3 else 1, box[1], box[0]):
        raise ValueError(f"mask shape {m.shape} does not match the box {box} of {c.name}")
    t, p = np.deg2rad(_f(theta, "H_theta")), np.deg2rad(_f(theta, "H_phi"))
    h = None
    if H:
        h = {1: (H,), 2: (H * np.cos(p), H * np.sin(p)),
             3: (H * np.sin(t) * np.cos(p), H * np.sin(t) * np.sin(p), H * np.cos(t))}[n]
        h = tuple(float(v) for v in h)
    kw = dict(extra=_dmi(c, D) if (n == 3 and D) else {}, h=h, cubic_k2=K2, mask=m)
    name = theta.get("name") or f"{model} {c.name}"

    def make(per):
        k1 = None
        if n == 3 and (K1 or Ks or K2):
            mm = np.ones((c.b, box[2] if c.D == 3 else 1, box[1], box[0]), bool) if m is None else m
            k1 = (np.where(_surface(c, box, per, mm), Ks, K1) * mm).astype(np.float32)
        return System(str(name), n, c, box, per, J, cubic_k1=k1, **kw)
    try:
        return make(periodic)
    except ValueError as e:
        if not any(periodic) or "colouring" not in str(e):
            raise
        (log or print)(f"  no admissible colouring for the periodic box {box}: using open boundaries")
        return make((False,) * c.D)


def temperature_ladder(T_max: float, T_min: float, T_step: float) -> np.ndarray:
    """Descending temperatures T_max, T_max - T_step, ..., T_min."""
    T_max, T_min, T_step = float(T_max), float(T_min), float(T_step)
    if T_step <= 0 or T_min <= 0 or T_max < T_min:
        raise ValueError("need T_max >= T_min > 0 and T_step > 0")
    k = int(np.floor((T_max - T_min) / T_step + 1e-9))
    return np.round(T_max - T_step * np.arange(k + 1), 10)


def size_estimate_gb(n_states: int, replicas: int, n_saved: int, n_sites: int) -> float:
    """Size of configurations.npy (float32, three components)."""
    return n_states * replicas * n_saved * n_sites * 3 * 4 / 1e9


# ------------------------------------------------------------------ one state
def _spins(A: Sampler, S: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    """(B, n_active, 3) float32 spins of the active sites, zero third (and second) components padded."""
    x = A.unpack(S).float().permute(0, 2, 3, 4, 5, 1).cpu().numpy()   # (B, b, L3, L2, L1, n)
    x = x[:, mask]
    out = np.zeros(x.shape[:2] + (3,), np.float32)
    out[..., :x.shape[-1]] = x
    return out


def simulate_state(s: System, replicas: int, temps, n_therm: int, n_meas: int, every: int, config_every: int,
                   seed: int, device="cpu", kB: float = 1.0, log=print) -> dict:
    """Run the ladder for one state. Energies are totals; M is the mean magnetization per site."""
    A = Sampler(s, int(replicas), device, seed)
    S = A.init_spins(seed=seed)
    mask = s.site_mask()
    n_act = float(s.n_active)
    k = max(1, int(every))
    m = max(1, int(n_meas) // k)
    ce = max(1, int(config_every))
    temps = np.asarray(temps, float)
    nT, B = len(temps), int(replicas)
    saved = [t for t in range(nT) if t % ce == 0]
    out = dict(M=np.zeros((B, nT, 3), np.float32), E=np.zeros((B, nT)), E2=np.zeros((B, nT)),
               Mabs=np.zeros((B, nT)), M2=np.zeros((B, nT)),
               configs=np.zeros((B, len(saved), int(mask.sum()), 3), np.float32),
               config_temperatures=temps[saved])

    def observe():
        E = A.energy(S)
        Mv = A.magnetization(S).sum(dim=1) / n_act               # (B, n)
        return E, Mv

    t0 = time.perf_counter()
    for t, T in enumerate(temps):
        A.set_temperature(kB * float(T))
        if n_therm > 0:
            A.sweep(S, int(n_therm))
        E_ref = observe()[0]
        dE = torch.zeros_like(E_ref)
        dE2, Ma, M2 = torch.zeros_like(E_ref), torch.zeros_like(E_ref), torch.zeros_like(E_ref)
        Mv_acc = None
        for _ in range(m):
            A.sweep(S, k)
            E, Mv = observe()
            d = E - E_ref
            dE += d
            dE2 += d * d
            a = Mv.norm(dim=1)
            Ma += a
            M2 += a * a
            Mv_acc = Mv.clone() if Mv_acc is None else Mv_acc + Mv
        dE, dE2 = dE / m, dE2 / m
        out["E"][:, t] = (E_ref + dE).cpu().numpy()
        out["E2"][:, t] = (dE2 + 2 * E_ref * dE + E_ref * E_ref).cpu().numpy()
        out["Mabs"][:, t] = (Ma / m).cpu().numpy()
        out["M2"][:, t] = (M2 / m).cpu().numpy()
        out["M"][:, t, :s.n] = (Mv_acc / m).float().cpu().numpy()
        if t in saved:
            out["configs"][:, saved.index(t)] = _spins(A, S, mask)
        if t % max(1, nT // 10) == 0 or t == nT - 1:
            log(f"    T = {T:9.4f}   <E>/site = {out['E'][:, t].mean() / n_act:+.4f}"
                f"   <|M|> = {out['Mabs'][:, t].mean():.4f}   {time.perf_counter() - t0:8.1f} s")
    return out


# ------------------------------------------------------------------ a run
def default_config() -> dict:
    """Every setting of a run, with working defaults (a small sc Heisenberg ferromagnet)."""
    return dict(
        model="heisenberg",               # ising | xy | heisenberg
        crystal="sc",                     # square triangular honeycomb kagome (2D) | sc bcc fcc (3D)
        box=[12, 12, 12],                 # cells along x, y[, z]
        periodic=[True, True, True],
        coords="",                        # "" | cartesian | spherical | cylindrical
        shape="",                         # condition in those coordinates; "" = every site active
        states=[{"name": "J=1", "J": {"1": 1.0}}],
        units="reduced",                  # reduced: J and T in units of J (kB = 1) | meV: T in kelvin
        T_max=2.0, T_min=0.2, T_step=0.1,
        thermalization=1000, measurement=2000, measure_every=10,
        replicas=32,
        config_every=1,                   # save every replica's configuration every this many T steps
        seed=1,
    )


def _kB(units: str) -> float:
    u = str(units).strip().lower()
    if u not in ("reduced", "mev"):
        raise ValueError("units must be 'reduced' (kB = 1) or 'meV' (T in kelvin)")
    return 1.0 if u == "reduced" else KB_MEV


def prepare(cfg: dict, out, log=print) -> dict:
    """Validate a configuration, write run.json and sites.npy, and allocate configurations.npy."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = {**default_config(), **cfg}
    if not run["states"]:
        raise ValueError("define at least one state")
    kB = _kB(run["units"])
    temps = temperature_ladder(run["T_max"], run["T_min"], run["T_step"])
    pos, mask = active_sites(run["crystal"], run["box"], run["coords"], run["shape"])
    s0 = build_system(run["model"], run["crystal"], run["box"], run["periodic"], run["states"][0], mask, log)
    ce = max(1, int(run["config_every"]))
    n_saved = len(range(0, len(temps), ce))
    n_sites = int(mask.sum())
    run.update(kB=kB, temperatures=temps.tolist(), periodic_used=list(s0.periodic), n_sites=n_sites,
               n_saved=n_saved, config_temperatures=temps[::ce].tolist())
    for stale in out.glob("state_*.npz"):          # a new run never mixes with an earlier one's blocks
        stale.unlink()
    (out / "run.json").write_text(json.dumps(run, indent=1))
    np.save(out / "sites.npy", pos[mask].astype(np.float32))
    np.save(out / "mask.npy", mask)
    shape = (len(run["states"]), int(run["replicas"]), n_saved, n_sites, 3)
    np.lib.format.open_memmap(out / "configurations.npy", mode="w+", dtype=np.float32, shape=shape).flush()
    gb = size_estimate_gb(*shape[:4])
    log(f"{len(run['states'])} state(s) x {run['replicas']} replicas x {len(temps)} temperatures, "
        f"{n_sites} active sites; configurations.npy = {gb:.3f} GB")
    return run


def replica_blocks(replicas: int, parts: int) -> list[tuple[int, int]]:
    """Split replicas 0..R-1 into at most `parts` contiguous blocks whose sizes differ by at most one."""
    R, g = int(replicas), max(1, min(int(parts), int(replicas)))
    q, r = divmod(R, g)
    edges = [0]
    for k in range(g):
        edges.append(edges[-1] + q + (1 if k < r else 0))
    return list(zip(edges[:-1], edges[1:]))


def run(cfg: dict, state_indices, out, device="cpu", log=print, replicas=None, part: int = 0) -> None:
    """Simulate the given states of a prepared run for the replica block [r0, r1) (all replicas by default).

    Each block is an independent set of chains: its seed is offset by the block index `part`, so blocks
    run on different devices never repeat one another. Writes state_XXX_pYY.npz and the block's slice of
    configurations.npy.
    """
    out = Path(out)
    r0, r1 = (0, int(cfg["replicas"])) if replicas is None else (int(replicas[0]), int(replicas[1]))
    if not 0 <= r0 < r1 <= int(cfg["replicas"]):
        raise ValueError(f"replica block [{r0}, {r1}) is outside 0..{cfg['replicas']}")
    mask = np.load(out / "mask.npy")
    cf = np.load(out / "configurations.npy", mmap_mode="r+")
    for i in state_indices:
        theta = cfg["states"][i]
        log(f"state {i} ({theta.get('name', i)}), replicas {r0}-{r1 - 1}, on {device}")
        s = build_system(cfg["model"], cfg["crystal"], cfg["box"], cfg["periodic_used"], theta, mask, log)
        r = simulate_state(s, r1 - r0, cfg["temperatures"], int(cfg["thermalization"]),
                           int(cfg["measurement"]), int(cfg["measure_every"]), int(cfg["config_every"]),
                           int(cfg["seed"]) + 1000 * int(i) + 100000 * int(part), device, float(cfg["kB"]), log)
        cf[i, r0:r1] = r.pop("configs")
        cf.flush()
        np.savez(out / f"state_{i:03d}_p{int(part):02d}.npz", r0=r0, r1=r1, **r)
    del cf


def _state_parts(out: Path, i: int, R: int) -> list:
    """The replica blocks of state i, in replica order; they must cover 0..R-1 exactly once."""
    parts = sorted((np.load(f) for f in out.glob(f"state_{i:03d}_p*.npz")), key=lambda z: int(z["r0"]))
    edge = 0
    for z in parts:
        if int(z["r0"]) != edge:
            break
        edge = int(z["r1"])
    if edge != R or not parts:
        raise ValueError(f"state {i}: its replica blocks do not cover replicas 0..{R - 1} exactly once")
    return parts


def collect(out) -> dict:
    """Combine the state files: arrays (n_states, replicas, nT[, 3]) and Cv, chi per site (n_states, nT)."""
    out = Path(out)
    cfg = json.loads((out / "run.json").read_text())
    T = np.asarray(cfg["temperatures"])
    N, kB = float(cfg["n_sites"]), float(cfg["kB"])
    R = int(cfg["replicas"])
    states = [_state_parts(out, i, R) for i in range(len(cfg["states"]))]
    res = {k: np.stack([np.concatenate([z[k] for z in parts]) for parts in states])
           for k in ("M", "E", "E2", "Mabs", "M2")}
    varE = np.maximum(res["E2"] - res["E"] ** 2, 0.0)
    varM = np.maximum(res["M2"] - res["Mabs"] ** 2, 0.0)
    combined = dict(temperatures=T, magnetization=res["M"], energy=res["E"], energy2=res["E2"],
                    Mabs=res["Mabs"], M2=res["M2"],
                    Cv=(varE / (kB * T ** 2 * N)).mean(axis=1), chi=(N * varM / (kB * T)).mean(axis=1),
                    E_site=(res["E"] / N).mean(axis=1), M_site=res["Mabs"].mean(axis=1),
                    state_names=np.array([str(s.get("name", i)) for i, s in enumerate(cfg["states"])]))
    np.save(out / "magnetization.npy", combined["magnetization"])
    np.savez(out / "results.npz", **combined)
    return combined


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory holding run.json")
    ap.add_argument("--states", required=True, help="comma-separated state indices")
    ap.add_argument("--replicas", default=None, help="replica block r0:r1 (default: every replica)")
    ap.add_argument("--part", type=int, default=0, help="index of the replica block; offsets its seed")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = json.loads((Path(a.run) / "run.json").read_text())
    block = None if a.replicas is None else tuple(int(x) for x in a.replicas.split(":"))
    run(cfg, [int(x) for x in a.states.split(",") if x != ""], a.run, a.device, lambda m: print(m, flush=True),
        replicas=block, part=a.part)


if __name__ == "__main__":
    main()
