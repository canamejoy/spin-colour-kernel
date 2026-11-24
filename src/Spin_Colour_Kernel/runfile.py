"""Write a ready-to-run notebook (.ipynb) or Python script (.py) for one run configuration.

The files are the ones the SCK configurator page (ready_for_run/configurator.html) downloads: the shipped
simulation notebook with only the values of its parameters cell rewritten. The template travels with the
package in data/run_template.json, which notebooks/build_configurator.py regenerates.

A configuration is a dict with the keys of DEFAULTS; missing keys take the default value:

    model        "ising" | "xy" | "heisenberg"
    units        "reduced" (J, k_B = 1) | "meV" (meV, kelvin)
    crystal      "square" "triangular" "honeycomb" "kagome" (2D) | "sc" "bcc" "fcc" (3D)
    box          unit cells per axis, [Lx, Ly] or [Lx, Ly, Lz]
    periodic     one bool per axis
    coords       "" (every site) | "cartesian" | "spherical" | "cylindrical"
    shape        condition in those coordinates, e.g. "(rho <= 10) & (abs(z) <= 2)"
    states       list of {name, J: {shell: value}, D, K1, Ks, K2, H, H_theta, H_phi}
    T_max, T_min, T_step, thermalization, measurement, measure_every, replicas, seed, config_every,
    output_dir   folder the run writes to
    gpus         "all" or a list of device indices
"""

from __future__ import annotations

import copy
import json
import math
import re
from functools import lru_cache
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent / "data" / "run_template.json"
STATE_KEYS = ("D", "K1", "Ks", "K2", "H", "H_theta", "H_phi")
FORMATS = ("ipynb", "py")


@lru_cache(maxsize=1)
def template() -> dict:
    """The notebook template: notebook, paramsCell, params, script, paramsMarker, defaults, commit."""
    return json.loads(TEMPLATE_PATH.read_text())


def defaults() -> dict:
    return copy.deepcopy(template()["defaults"])


def config(cfg: dict | None = None) -> dict:
    """The configuration with every missing key taken from the defaults (the shipped notebook's values)."""
    out = defaults()
    for k, v in (cfg or {}).items():
        if k not in out:
            raise ValueError(f"unknown setting {k!r}; the settings are: {', '.join(out)}")
        out[k] = copy.deepcopy(v)
    return out


def _num(v) -> str:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {v!r}")
    return f"{x:.1f}" if x.is_integer() else repr(x)


def _int(v) -> str:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {v!r}")
    return str(math.trunc(x))


def _str(s) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _blank(v) -> bool:
    return v is None or v == ""


def _state(st: dict, i: int) -> str:
    parts = [f"name={_str(st.get('name') or f'state-{i + 1}')}"]
    J = [(s, v) for s, v in (st.get("J") or {"1": 1}).items() if not _blank(v)]
    parts.append("J={" + ", ".join(f"{_int(s)}: {_num(v)}" for s, v in J) + "}")
    parts += [f"{k}={_num(st[k])}" for k in STATE_KEYS if not _blank(st.get(k))]
    return f"    dict({', '.join(parts)}),"


def write_params(params: str, cfg: dict) -> str:
    """The parameters cell with its values replaced by cfg's; every comment and example line stays."""
    gpus = cfg.get("gpus", "all")
    values = dict(
        MODEL=_str(cfg["model"]), UNITS=_str(cfg["units"]), CRYSTAL=_str(cfg["crystal"]),
        BOX="[" + ", ".join(_int(x) for x in cfg["box"]) + "]",
        PERIODIC="[" + ", ".join("True" if p else "False" for p in cfg["periodic"]) + "]",
        COORDS=_str(cfg.get("coords") or ""), SHAPE=_str(cfg.get("shape") or ""),
        T_MAX=_num(cfg["T_max"]), T_MIN=_num(cfg["T_min"]), T_STEP=_num(cfg["T_step"]),
        THERMALIZATION=_int(cfg["thermalization"]), MEASUREMENT=_int(cfg["measurement"]),
        MEASURE_EVERY=_int(cfg["measure_every"]), REPLICAS=_int(cfg["replicas"]), SEED=_int(cfg["seed"]),
        CONFIG_EVERY=_int(cfg["config_every"]), OUTPUT_DIR=_str(cfg.get("output_dir") or "simulation_output"),
        GPUS='"all"' if gpus in (None, "all") else "[" + ", ".join(_int(g) for g in gpus) + "]",
    )
    lines, out, seen, i = params.split("\n"), [], set(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("STATES = ["):
            out.append("STATES = [")
            out += [_state(st, k) for k, st in enumerate(cfg["states"])]
            while i < len(lines) and lines[i] != "]":
                if re.match(r"^\s+#", lines[i]):       # the commented examples stay
                    out.append(lines[i])
                i += 1
            out.append("]")
            seen.add("STATES")
            i += 1
            continue
        m = re.match(r"^([A-Z_]+)(\s*=\s*)(.*?)(\s+#.*)?$", line)
        if m and m.group(1) in values:
            out.append(m.group(1) + m.group(2) + values[m.group(1)] + (m.group(4) or ""))
            seen.add(m.group(1))
        else:
            out.append(line)
        i += 1
    missing = [k for k in [*values, "STATES"] if k not in seen]
    if missing:
        raise ValueError(f"the template has no line for {', '.join(missing)}")
    return "\n".join(out)


def notebook(cfg: dict) -> str:
    """The .ipynb text for cfg (a full configuration, see config())."""
    t = template()
    nb = copy.deepcopy(t["notebook"])
    src = write_params(t["params"], cfg)
    nb["cells"][t["paramsCell"]]["source"] = src.splitlines(keepends=True)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def script(cfg: dict) -> str:
    """The .py text for cfg (a full configuration, see config())."""
    t = template()
    return t["script"].replace(t["paramsMarker"], write_params(t["params"], cfg), 1)


def render(cfg: dict, fmt: str = "ipynb") -> str:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    return notebook(cfg) if fmt == "ipynb" else script(cfg)


def write(cfg: dict, path, fmt: str | None = None) -> Path:
    """Write the run file for cfg to path; the format comes from fmt or from the path's extension."""
    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".") or "ipynb"
    if path.suffix != f".{fmt}":
        path = path.with_name(path.name + f".{fmt}") if not path.suffix else path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cfg, fmt))
    return path
