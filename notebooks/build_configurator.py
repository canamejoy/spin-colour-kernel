"""Build the run configurator: one self-contained HTML page that writes a ready-to-run notebook or Python script.

    python notebooks/build_configurator.py

The page (ready_for_run/configurator.html) needs no installation and no network. It carries the notebook that
build_simulation_notebook.py builds, with its parameters cell as a template, and a Python-script version of the
same cells; the form only rewrites the values of that cell, so the files it writes run exactly the code of the
shipped notebook. The preview (lattice, active sites, surface, colouring) runs core.cjs, a port of the package's
geometry and colouring that tests/test_configurator.py checks against the package itself.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_simulation_notebook as bsn  # noqa: E402
from build_nanodisk_notebook import ROOT, commit  # noqa: E402

HERE = Path(__file__).resolve().parent / "configurator"
OUT = ROOT / "ready_for_run" / "configurator.html"
MARKER = "#<<PARAMETERS>>"
PARAMS_CELL = 2

# The values of the shipped notebook's parameters cell, as the form holds them.
DEFAULTS = dict(model="heisenberg", units="reduced", crystal="sc", box=[12, 12, 12], periodic=[True, True, True],
                coords="", shape="",
                states=[dict(name="ferromagnet", J={"1": 1.0}), dict(name="with field", J={"1": 1.0}, H=0.2, H_theta=0.0)],
                T_max=2.0, T_min=0.2, T_step=0.1, thermalization=1000, measurement=2000, measure_every=10,
                replicas=32, seed=1, config_every=1, output_dir="simulation_output", gpus="all")


def _script(nb: dict) -> str:
    """The notebook's cells as one Python script; the parameters cell becomes MARKER."""
    out = [f'"""Spin Monte Carlo run written by the spin-colour-kernel configurator (commit {commit()}).\n\n'
           "Run it with  python run_simulation.py  on a machine with an NVIDIA GPU (or the CPU, slowly). It writes\n"
           "the package modules next to itself, checks each against its sha256, runs every state and saves the\n"
           'results and summary.png in OUTPUT_DIR. Edit only section 1."""\n']
    for i, cell in enumerate(nb["cells"][1:], start=1):
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            out.append("# %% " + src.splitlines()[0].lstrip("# ").strip())
        elif i == PARAMS_CELL:
            out.append(MARKER)
        else:
            out.append(src.replace("plt.tight_layout(); plt.show()",
                                   'plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/summary.png", dpi=150); plt.show()'))
    return "\n\n".join(out) + "\n"


def template() -> dict:
    nb = bsn.build()
    params = "".join(nb["cells"][PARAMS_CELL]["source"])
    if not params.lstrip().startswith("# ═"):
        raise SystemExit(f"cell {PARAMS_CELL} of the notebook is not the parameters cell")
    return dict(notebook=nb, paramsCell=PARAMS_CELL, params=params, script=_script(nb), paramsMarker=MARKER,
                defaults=DEFAULTS, commit=commit())


def build() -> str:
    page = (HERE / "index.html").read_text()
    data = json.dumps(template(), ensure_ascii=False).replace("</", "<\\/")
    parts = {"STYLE": (HERE / "style.css").read_text(), "CORE": (HERE / "core.cjs").read_text(),
             "APP": (HERE / "app.js").read_text(), "TEMPLATE": data}
    for k, v in parts.items():
        token = f"/*{{{{{k}}}}}*/"
        if page.count(token) != 1:
            raise SystemExit(f"index.html must hold {token} exactly once")
        page = page.replace(token, v)
    if re.search(r"/\*\{\{\w+\}\}\*/", page):
        raise SystemExit("unfilled slot left in the page")
    return page


if __name__ == "__main__":
    OUT.write_text(build())
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
