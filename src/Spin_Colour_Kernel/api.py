"""The short Python interface: describe, check, write and run a simulation, then read and plot its results.

    import sck                                    # or: import Spin_Colour_Kernel as sck

    sck.describe()                                # models, crystals, parameters, defaults
    sck.check(crystal="sc", box=[16, 16, 16])     # errors, sites, colouring, size, without running
    res = sck.run(crystal="sc", box=[16, 16, 16],
                  states=[dict(name="ferromagnet", J={1: 1.0})],
                  T_max=2.0, T_min=0.5, T_step=0.1)
    res.plot()                                    # E, |M|, Cv and chi against T
    res.table()                                   # the same numbers as rows

A configuration is a dict (or keyword arguments) with the settings that `describe()["defaults"]` lists;
every setting left out takes its default. `run` uses every NVIDIA GPU with compute capability 7.0 or more,
one process per GPU with an equal block of the replicas of every state, or the CPU through the Triton
interpreter when there is none (slow: keep CPU runs small). It writes the same output folder as the
ready-to-run notebook and returns its results.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import agent, runfile

__all__ = ["describe", "check", "write", "run", "load", "Results"]

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[1])


def describe() -> dict:
    """The models, crystals, coordinate systems, state parameters, units and every default setting."""
    return agent.describe()


def _merge(config: dict | None, settings: dict) -> dict:
    cfg = dict(config or {})
    cfg.update(settings)
    return cfg


def check(config: dict | None = None, **settings) -> dict:
    """Validate a configuration without running it; never raises for a bad one (errors come back)."""
    return agent.check(_merge(config, settings))


def write(path="spin_simulation.ipynb", config: dict | None = None, fmt: str | None = None, **settings) -> Path:
    """Write the run as a ready-to-run notebook (.ipynb) or script (.py); refuses a configuration with errors."""
    r = agent.check(_merge(config, settings))
    if not r["ok"]:
        raise ValueError("the configuration has errors: " + "; ".join(r["errors"]))
    return runfile.write(r["config"], path, fmt)


def _devices(gpus) -> list[int]:
    import torch
    if not torch.cuda.is_available():
        return []
    usable = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i) >= (7, 0)]
    if gpus not in (None, "all"):
        wanted = [gpus] if isinstance(gpus, int) else list(gpus)
        usable = [g for g in usable if g in wanted]
    return usable


def _simulation_config(full: dict) -> dict:
    keys = ("model", "crystal", "box", "periodic", "coords", "shape", "states", "units", "T_max", "T_min", "T_step",
            "thermalization", "measurement", "measure_every", "replicas", "config_every", "seed")
    cfg = {k: full[k] for k in keys}
    states = []
    for i, s in enumerate(cfg["states"]):
        s = dict(s)
        s.setdefault("name", f"state-{i + 1}")
        s["J"] = {str(k): float(v) for k, v in (s.get("J") or {1: 1.0}).items()}
        states.append(s)
    cfg["states"] = states
    return cfg


def run(config: dict | None = None, *, output_dir=None, gpus=None, verbose: bool = True, **settings) -> "Results":
    """Run a simulation on every usable GPU (or the CPU) and return its results.

    config and keyword settings are merged (keywords win); output_dir and gpus override the configuration's
    own `output_dir` and `gpus`. The run is checked first and refused, before anything starts, when it has errors.
    """
    cfg = _merge(config, settings)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if gpus is not None:
        cfg["gpus"] = gpus
    r = agent.check(cfg)
    if not r["ok"]:
        raise ValueError("the configuration has errors: " + "; ".join(r["errors"]))
    full = r["config"]
    say = print if verbose else (lambda *a, **k: None)
    for w in r["warnings"]:
        say("warning:", w)
    usable = _devices(full.get("gpus", "all"))
    if not usable:
        os.environ["TRITON_INTERPRET"] = "1"      # kernels compiled after this point run on the CPU
    from . import simulate as SIM
    out = Path(full.get("output_dir") or "simulation_output")
    sim_cfg = _simulation_config(full)
    names = [s["name"] for s in sim_cfg["states"]]
    if len(set(names)) != len(names):
        raise ValueError("state names must be distinct")
    log = say if verbose else (lambda *a, **k: None)
    prepared = SIM.prepare(sim_cfg, out, log=log)
    say(f"{len(prepared['temperatures'])} temperatures from {prepared['temperatures'][0]} to "
        f"{prepared['temperatures'][-1]}; " + (f"{len(usable)} GPU(s): {usable}" if usable else
                                               "no usable GPU: CPU through the Triton interpreter (slow)"))
    t0 = time.perf_counter()
    if not usable:
        SIM.run(prepared, list(range(len(names))), out, device="cpu", log=log)
    else:
        _run_on_gpus(out, len(names), int(full["replicas"]), usable, say)
    say(f"total: {(time.perf_counter() - t0) / 60:.2f} min")
    return load(out)


def _run_on_gpus(out: Path, n_states: int, replicas: int, usable: list[int], say) -> None:
    from . import simulate as SIM
    blocks = SIM.replica_blocks(replicas, len(usable))
    every = ",".join(map(str, range(n_states)))
    procs = []
    for part, (g, (r0, r1)) in enumerate(zip(usable, blocks)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   PYTHONPATH=os.pathsep.join(p for p in (_PACKAGE_ROOT, os.environ.get("PYTHONPATH", "")) if p))
        env.pop("TRITON_INTERPRET", None)
        cmd = [sys.executable, "-u", "-m", "Spin_Colour_Kernel.simulate", "--run", str(out), "--states", every,
               "--replicas", f"{r0}:{r1}", "--part", str(part), "--device", "cuda:0"]
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        say(f"GPU {g}: replicas {r0}-{r1 - 1} of every state")

    def pump(g, p):
        for line in p.stdout:
            say(f"GPU {g} {line}", end="", flush=True)
    threads = [threading.Thread(target=pump, args=gp) for gp in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [g for g, p in procs if p.wait() != 0]
    if bad:
        raise RuntimeError(f"the run failed on GPU(s) {bad}: read the lines printed above")


def load(output_dir="simulation_output") -> "Results":
    """The results of a finished run, read from its output folder (it combines the per-GPU files once)."""
    import json
    from . import simulate as SIM
    out = Path(output_dir)
    data = SIM.collect(out)
    return Results(out, json.loads((out / "run.json").read_text()), data)


class Results:
    """A finished run: per state and temperature, E and |M| per site, Cv and chi, and the output folder.

    Attributes: output_dir, config (run.json), names, temperatures, and the arrays energy_per_site,
    magnetization_per_site, specific_heat and susceptibility, each (n_states, n_temperatures).
    `arrays` holds everything results.npz holds; `configurations()` opens the saved spins without loading them.
    """

    def __init__(self, output_dir: Path, config: dict, data: dict):
        self.output_dir, self.config, self.arrays = Path(output_dir), config, data
        self.names = [str(n) for n in data["state_names"]]
        self.temperatures = data["temperatures"]
        self.energy_per_site = data["E_site"]
        self.magnetization_per_site = data["M_site"]
        self.specific_heat = data["Cv"]
        self.susceptibility = data["chi"]

    @property
    def temperature_unit(self) -> str:
        return "K" if str(self.config.get("units", "")).lower() == "mev" else "J/k_B"

    def configurations(self):
        """The saved spin configurations, memory-mapped: (states, replicas, saved T, sites, components)."""
        import numpy as np
        return np.load(self.output_dir / "configurations.npy", mmap_mode="r")

    def table(self) -> list[dict]:
        """One row per state and temperature."""
        return [dict(state=n, T=float(t), E_site=float(self.energy_per_site[i, k]),
                     M_site=float(self.magnetization_per_site[i, k]), Cv=float(self.specific_heat[i, k]),
                     chi=float(self.susceptibility[i, k]))
                for i, n in enumerate(self.names) for k, t in enumerate(self.temperatures)]

    def plot(self, path=None, show: bool = True):
        """E, |M|, Cv and chi per site against T for every state; saved to path when one is given."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        for i, n in enumerate(self.names):
            for a, y, label in ((ax[0, 0], self.energy_per_site, "energy per site"),
                                (ax[0, 1], self.magnetization_per_site, "|M| per site"),
                                (ax[1, 0], self.specific_heat, "Cv per site"), (ax[1, 1], self.susceptibility, "χ")):
                a.plot(self.temperatures, y[i], "o-", ms=3, lw=1.2, label=n)
                a.set_xlabel(f"T ({self.temperature_unit})")
                a.set_ylabel(label)
                a.grid(alpha=0.25)
        ax[0, 0].legend(fontsize=9)
        c = self.config
        fig.suptitle(f"{c['model']} on {c['crystal']} {list(c['box'])} · {len(self.names)} state(s) · "
                     f"{c['replicas']} replicas", y=1.0)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150)
        if show:
            plt.show()
        return fig

    def __repr__(self) -> str:
        T = self.temperatures
        rows = [f"Results({self.output_dir}): {len(self.names)} state(s), {len(T)} temperatures "
                f"{T[0]:g} to {T[-1]:g} {self.temperature_unit}"]
        for i, n in enumerate(self.names):
            rows.append(f"  {n}: at T = {T[-1]:g}  E/site {self.energy_per_site[i, -1]:.5f}  "
                        f"|M| {self.magnetization_per_site[i, -1]:.4f}  Cv {self.specific_heat[i, -1]:.4f}  "
                        f"χ {self.susceptibility[i, -1]:.4f}")
        return "\n".join(rows)
