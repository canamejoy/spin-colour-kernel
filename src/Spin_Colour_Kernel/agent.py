"""What an agent needs to prepare a run: the vocabulary of a configuration, and a check that reports instead of raising.

    describe()   the models, crystals, coordinate systems, state parameters and settings, with their defaults
    check(cfg)   the configuration completed with the defaults, its errors, and, when there are none, the active
                 sites, surface, colouring of every state, temperature ladder and the size of the run

The MCP server (Spin_Colour_Kernel.mcp) exposes both as tools; they are plain functions for any other caller.
"""

from __future__ import annotations

from . import crystals as cr
from . import runfile
from . import simulate as sim

CRYSTALS = {"square": 2, "triangular": 2, "honeycomb": 2, "kagome": 2, "sc": 3, "bcc": 3, "fcc": 3}
STATE_PARAMETERS = {
    "J": "exchange per neighbour shell, {shell: value} with shells 1 to 4; J > 0 is ferromagnetic",
    "D": "bulk-type (Bloch) DMI on in-plane nearest-neighbour bonds (Heisenberg only)",
    "K1": "cubic anisotropy on bulk sites (Heisenberg only)",
    "Ks": "cubic anisotropy on surface sites, fewer active nearest neighbours than the bulk; default K1 (Heisenberg only)",
    "K2": "second-order cubic anisotropy (Heisenberg only)",
    "H": "Zeeman field strength",
    "H_theta": "field polar angle in degrees, 0 is +z (Heisenberg)",
    "H_phi": "field azimuth in degrees (XY and Heisenberg)",
}
HAMILTONIAN = ("H = - sum_s J_s sum_<ij>_s S_i . S_j - sum_<ij>_1,in-plane S_i^T A_ij S_j "
               "+ sum_i K1_i (sx^2 sy^2 + sx^2 sz^2 + sy^2 sz^2) + K2 sum_i (sx sy sz)^2 - sum_i H . S_i, "
               "A_lm = D sum_k r_k eps_klm, K1_i = Ks on surface sites")


def describe() -> dict:
    return dict(
        hamiltonian=HAMILTONIAN,
        models={"ising": "spins s = +-1; J and H only", "xy": "unit spins in the plane; J and H only",
                "heisenberg": "unit spins in 3D; every parameter"},
        crystals={k: f"{d}D" for k, d in CRYSTALS.items()},
        coordinates={k: list(v) for k, v in sim.COORDS.items()},
        shape_functions=[k for k in sim._MATH if k != "pi"] + ["pi"],
        shape_rules="a condition on the coordinates, origin at the centre of the box, angles in radians; "
                    "combine comparisons with & | ^ ~ and parentheses, e.g. (rho <= 10) & (abs(z) <= 2)",
        state_parameters=STATE_PARAMETERS,
        units={"reduced": "energies in units of J, k_B = 1", "meV": "energies in meV, temperatures in kelvin"},
        defaults=runfile.defaults(),
        formats=list(runfile.FORMATS),
    )


def _int_list(v, name, n=None):
    if not isinstance(v, (list, tuple)) or (n is not None and len(v) != n):
        raise ValueError(f"{name} must be a list of {n} values" if n else f"{name} must be a list")
    return list(v)


def check(cfg: dict | None = None) -> dict:
    """Validate a configuration. Never raises for a bad configuration: errors come back in the result."""
    errors: list[str] = []
    try:
        full = runfile.config(cfg)
    except ValueError as e:
        return dict(ok=False, errors=[str(e)], config=None)
    out = dict(ok=False, errors=errors, warnings=[], config=full)
    model, crystal = str(full["model"]).lower(), str(full["crystal"])
    if model not in sim.MODELS:
        errors.append(f"unknown model {model!r}; known: {', '.join(sim.MODELS)}")
    if crystal not in CRYSTALS:
        errors.append(f"unknown crystal {crystal!r}; known: {', '.join(CRYSTALS)}")
    if errors:
        return out
    D = CRYSTALS[crystal]
    try:
        box = [int(x) for x in _int_list(full["box"], "box", D)]
        periodic = [bool(x) for x in _int_list(full["periodic"], "periodic", D)]
        if min(box) < 1:
            raise ValueError("every box size must be at least 1 cell")
    except (TypeError, ValueError) as e:
        errors.append(f"{e} (the {crystal} crystal is {D}D)")
        return out
    try:
        temps = sim.temperature_ladder(full["T_max"], full["T_min"], full["T_step"])
    except (TypeError, ValueError) as e:
        errors.append(str(e))
        temps = None
    for k in ("thermalization", "measurement", "measure_every", "replicas", "config_every"):
        try:
            if int(full[k]) < 1:
                errors.append(f"{k} must be at least 1")
        except (TypeError, ValueError):
            errors.append(f"{k} must be a whole number")
    if not full["states"]:
        errors.append("define at least one state")
    try:
        pos, mask = sim.active_sites(crystal, box, full["coords"], full["shape"])
    except (ValueError, KeyError) as e:
        errors.append(str(e))
        mask = None
    states = []
    for i, st in enumerate(full["states"]):
        name = st.get("name") or f"state-{i + 1}"
        if mask is None:
            break
        try:
            s = sim.build_system(model, crystal, box, periodic, st, mask, log=lambda m: None)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"state {name!r}: {e}")
            continue
        states.append(dict(name=name, colours=s.colouring.n_colours, colouring=s.colouring.describe(),
                           periodic_used=list(s.periodic)))
        if list(s.periodic) != periodic:
            out["warnings"].append(f"state {name!r}: no admissible colouring for the periodic box {box}; "
                                   "it runs with open boundaries")
    if errors:
        return out
    c = cr.make(crystal)
    n_sites = int(mask.sum())
    surface = int(sim._surface(c, box, periodic, mask).sum())
    n_states, R, nT = len(full["states"]), int(full["replicas"]), len(temps)
    n_saved = len(range(0, nT, max(1, int(full["config_every"]))))
    out.update(
        ok=True,
        sites=dict(box=box[0] * box[1] * (box[2] if D == 3 else 1) * c.b, active=n_sites, surface=surface,
                   bulk=n_sites - surface),
        states=states,
        temperatures=dict(count=nT, first=float(temps[0]), last=float(temps[-1])),
        budget=dict(chains=n_states * R, spin_updates=n_states * R * nT * n_sites
                    * (int(full["thermalization"]) + int(full["measurement"])),
                    configurations_saved_per_chain=n_saved,
                    output_gb=round(sim.size_estimate_gb(n_states, R, n_saved, n_sites), 4)),
    )
    return out
