"""The configurator's JavaScript core (notebooks/configurator/core.cjs) against the Python package it ports.

Every case runs the same input through both and compares: bond tables, colourings (periodic seam and the
open-border fallback included), active-site masks, surface masks, the temperature ladder, and the parameters
cell it writes, which Python must read back to the configuration it was given. Needs Node; skipped without it.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "notebooks" / "configurator" / "core.cjs"
sys.path.insert(0, str(ROOT / "src"))

from Spin_Colour_Kernel import colouring as col  # noqa: E402
from Spin_Colour_Kernel import crystals as cr  # noqa: E402
from Spin_Colour_Kernel import simulate as SIM  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DRIVER = r"""
const SCK = require(process.argv[1]);
const req = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = req.map((q) => {
  try {
    if (q.op === "bonds") return SCK.makeCrystal(q.crystal, q.shells).bonds.map((b) => [b.alpha, b.alpha2, b.delta, b.shell]);
    if (q.op === "search") { const c = SCK.search(SCK.makeCrystal(q.crystal, q.shells), q.box, q.periodic); return c && [c.m, c.K, c.o]; }
    if (q.op === "analyse") {
      const r = SCK.analyse(q.cfg);
      return { errors: r.errors, warnings: r.warnings, mask: r.mask && Array.from(r.mask), surface: r.surface && Array.from(r.surface),
               colour: r.colour && Array.from(r.colour), periodicUsed: r.periodicUsed, nColours: r.nColours,
               temperatures: r.temperatures, colouring: r.colouring && [r.colouring.m, r.colouring.K, r.colouring.o] };
    }
    if (q.op === "params") return SCK.writeParams(q.template, q.cfg);
    if (q.op === "verify") return SCK.verifyOnBox({ m: q.m, K: q.K, o: q.o }, SCK.makeCrystal(q.crystal, q.shells), q.box, q.periodic);
    if (q.op === "series") return SCK.series(q.a, q.b, q.s);
    if (q.op === "sweep") return SCK.sweepStates(q.base, q.rows, q.prefix, q.limit);
    if (q.op === "files") { const t = JSON.parse(require("fs").readFileSync(q.tpl, "utf8")); return [SCK.notebookFor(t, q.cfg), SCK.scriptFor(t, q.cfg)]; }
  } catch (e) { return { thrown: e.message }; }
});
process.stdout.write(JSON.stringify(out));
"""


def js(requests):
    r = subprocess.run([NODE, "-e", DRIVER, str(CORE)], input=json.dumps(requests), capture_output=True, text=True,
                       check=True)
    return json.loads(r.stdout)


CASES = [(c, s) for c in ("square", "triangular", "honeycomb", "kagome", "sc", "bcc", "fcc") for s in ((1,), (1, 2))]
CASES += [("sc", (1, 2, 3)), ("sc", (1, 2, 3, 4)), ("square", (1, 3))]


def test_bond_tables_match_the_package():
    got = js([{"op": "bonds", "crystal": c, "shells": list(s)} for c, s in CASES])
    for (c, s), bonds in zip(CASES, got):
        want = [[b.alpha, b.alpha2, list(b.delta), b.shell] for b in cr.make(c, s).bonds]
        assert bonds == want, (c, s)


BOXES = [("square", (1,), [8, 8], [True, True]), ("square", (1, 2), [6, 6], [True, True]),
         ("square", (1, 2), [7, 6], [True, True]), ("triangular", (1,), [9, 9], [True, True]),
         ("triangular", (1,), [8, 8], [True, True]), ("honeycomb", (1,), [6, 6], [True, True]),
         ("kagome", (1,), [6, 6], [True, True]), ("sc", (1,), [5, 5, 5], [True, True, True]),
         ("sc", (1, 2, 3, 4), [8, 8, 8], [True, True, True]), ("bcc", (1, 2), [6, 6, 6], [True, True, True]),
         ("fcc", (1,), [4, 4, 4], [True, True, True]), ("sc", (1,), [5, 5, 5], [False, False, False])]


def test_colourings_match_the_package_seam_included():
    got = js([{"op": "search", "crystal": c, "shells": list(s), "box": b, "periodic": p} for c, s, b, p in BOXES])
    for (c, s, b, p), g in zip(BOXES, got):
        want = col.search(cr.make(c, s), tuple(b), tuple(p))
        assert g == (None if want is None else [list(want.m), [list(r) for r in want.K], [list(r) for r in want.o]]), (c, s, b)


def _cfg(**kw):
    base = dict(model="heisenberg", crystal="sc", box=[12, 12, 5], periodic=[False, False, False], coords="", shape="",
                states=[dict(name="a", J={1: 1.0})], T_max=2.0, T_min=0.2, T_step=0.1, replicas=4, config_every=1)
    base.update(kw)
    return base


SHAPES = [
    _cfg(),
    _cfg(coords="cylindrical", shape="(rho <= 5.2) & (abs(z) <= 2)"),
    _cfg(coords="spherical", shape="r <= 4"),
    _cfg(coords="spherical", shape="(r <= 5) & ~(theta < pi/4)"),
    _cfg(coords="cartesian", shape="(abs(x) + abs(y) <= 5) | (z > 1.5)"),
    _cfg(coords="cartesian", shape="mod(floor(x) + floor(y), 2) == 0"),
    _cfg(crystal="triangular", box=[14, 14], periodic=[False, False], coords="cylindrical", shape="rho <= 5"),
    _cfg(crystal="honeycomb", box=[10, 10], periodic=[True, True]),
    _cfg(crystal="kagome", box=[8, 8], periodic=[False, False], coords="cartesian", shape="x**2 + y**2 < 16"),
    _cfg(crystal="fcc", box=[5, 5, 5], periodic=[False, False, False], coords="spherical", shape="r <= 2.3"),
    _cfg(crystal="bcc", box=[6, 6, 6], periodic=[True, True, True], states=[dict(name="a", J={1: 1.0, 2: 0.3})]),
    _cfg(crystal="sc", box=[7, 7, 7], periodic=[True, True, True]),             # odd periodic box: open fallback
]


@pytest.mark.parametrize("i", range(len(SHAPES)))
def test_active_sites_surface_and_colouring_match_the_package(i):
    cfg = SHAPES[i]
    got = js([{"op": "analyse", "cfg": cfg}])[0]
    assert not got["errors"], got["errors"]
    _, mask = SIM.active_sites(cfg["crystal"], cfg["box"], cfg["coords"], cfg["shape"])
    assert got["mask"] == mask.astype(int).ravel().tolist()
    s = SIM.build_system(cfg["model"], cfg["crystal"], cfg["box"], cfg["periodic"], cfg["states"][0], mask, log=lambda *_: None)
    assert got["periodicUsed"] == list(s.periodic)
    assert bool(got["warnings"]) == (list(s.periodic) != list(cfg["periodic"]))
    c = s.colouring
    assert got["colouring"] == [list(c.m), [list(r) for r in c.K], [list(r) for r in c.o]]
    assert got["colour"] == s.colour_map().ravel().tolist()
    surf = SIM._surface(s.crystal, s.box, s.periodic, mask)
    assert got["surface"] == surf.astype(int).ravel().tolist()
    assert got["temperatures"] == SIM.temperature_ladder(cfg["T_max"], cfg["T_min"], cfg["T_step"]).tolist()


def test_shape_errors_are_reported_not_thrown():
    got = js([{"op": "analyse", "cfg": _cfg(coords="cylindrical", shape="r <= 3")},
              {"op": "analyse", "cfg": _cfg(coords="cylindrical", shape="rho <= 3 and z > 0")},
              {"op": "analyse", "cfg": _cfg(coords="cylindrical", shape="rho <= -1")},
              {"op": "analyse", "cfg": _cfg(model="ising", states=[dict(name="a", J={1: 1.0}, D=0.5)])}])
    assert [e["field"] for g in got for e in g["errors"]] == ["shape", "shape", "shape", "state.0.D"]
    assert "rho, phi, z" in got[0]["errors"][0]["msg"]
    assert "use & (and)" in got[1]["errors"][0]["msg"]


def _builder():
    spec = importlib.util.spec_from_file_location("bsn", ROOT / "notebooks" / "build_simulation_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "notebooks"))
    spec.loader.exec_module(mod)
    return mod


def _read_params(src):
    ns = {}
    exec(src, ns)
    return ns


FULL = dict(model="heisenberg", units="meV", crystal="sc", box=[40, 40, 5], periodic=[False, False, False],
            coords="cylindrical", shape='(rho <= 18.3) & (abs(z) <= 2)',
            states=[dict(name="disk A", J={1: 1.0, 2: 0.25}, D=0.88, K1=0.177, Ks=0.139, K2=0.034, H=0.483, H_theta=0),
                    dict(name='quote " B', J={1: -1.5})],
            T_max=20, T_min=0.1, T_step=0.1, thermalization=10000, measurement=10000, measure_every=10, replicas=50,
            seed=7, config_every=5, output_dir="out dir", gpus=[0, 1])


def test_the_parameters_cell_reads_back_to_the_configuration():
    tpl = _builder().PARAMS.strip("\n")
    src = js([{"op": "params", "template": tpl, "cfg": FULL}])[0]
    ns = _read_params(src)
    assert ns["MODEL"] == "heisenberg" and ns["UNITS"] == "meV" and ns["CRYSTAL"] == "sc"
    assert ns["BOX"] == [40, 40, 5] and ns["PERIODIC"] == [False, False, False]
    assert ns["COORDS"] == "cylindrical" and ns["SHAPE"] == FULL["shape"]
    assert ns["STATES"] == [dict(name="disk A", J={1: 1.0, 2: 0.25}, D=0.88, K1=0.177, Ks=0.139, K2=0.034, H=0.483,
                                 H_theta=0.0), dict(name='quote " B', J={1: -1.5})]
    assert (ns["T_MAX"], ns["T_MIN"], ns["T_STEP"]) == (20.0, 0.1, 0.1)
    assert (ns["THERMALIZATION"], ns["MEASUREMENT"], ns["MEASURE_EVERY"], ns["REPLICAS"], ns["SEED"]) == (10000, 10000, 10, 50, 7)
    assert (ns["CONFIG_EVERY"], ns["OUTPUT_DIR"], ns["GPUS"]) == (5, "out dir", [0, 1])
    # every comment line of the template survives
    assert [l for l in src.splitlines() if l.startswith("#")] == [l for l in tpl.splitlines() if l.startswith("#")]


def test_the_default_form_changes_only_the_parameters_cell_and_keeps_its_values(tmp_path):
    b = _builder()
    sys.path.insert(0, str(ROOT / "notebooks"))
    import build_configurator as bc
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps(bc.template()))
    nb, script = js([{"op": "files", "tpl": str(tpl), "cfg": bc.DEFAULTS}])[0]
    nb, ref = json.loads(nb), b.build()
    k = bc.PARAMS_CELL
    assert [c for i, c in enumerate(nb["cells"]) if i != k] == [c for i, c in enumerate(ref["cells"]) if i != k]
    got, want = _read_params("".join(nb["cells"][k]["source"])), _read_params("".join(ref["cells"][k]["source"]))
    assert {x: got[x] for x in want if x.isupper()} == {x: want[x] for x in want if x.isupper()}
    assert "# dict(name=\"nanodisk\"" in "".join(nb["cells"][k]["source"])
    compile(script, "run_simulation.py", "exec")
    head = script.split("# %% 2 · Hardware")[0]
    assert {x: v for x, v in _read_params(head).items() if x.isupper()} == {x: want[x] for x in want if x.isupper()}
    assert script.count("sha256 mismatch") == len(b.MODULES) and "summary.png" in script


def _same_build(shipped: str, fresh: str) -> bool:
    """Equal up to the commit hash they carry, which changes with every commit after the build."""
    sys.path.insert(0, str(ROOT / "notebooks"))
    import build_configurator as bc
    now = bc.commit()
    old = json.loads(shipped)["commit"] if shipped.lstrip().startswith("{") else \
        shipped.split('"commit": "')[1].split('"')[0]
    return shipped.replace(old, now) == fresh


def test_the_ready_for_run_page_is_the_current_build():
    sys.path.insert(0, str(ROOT / "notebooks"))
    import build_configurator as bc
    assert _same_build((ROOT / "ready_for_run" / "configurator.html").read_text(), bc.build())


def test_nothing_covers_the_model_canvas_when_the_message_is_hidden():
    css = (ROOT / "notebooks" / "configurator" / "style.css").read_text()
    assert ".view-msg[hidden] { display: none; }" in css and "pointer-events: none" in css.split(".view-msg {")[1].split("}")[0]


def test_a_parameter_sweep_lists_every_value_from_start_to_end():
    got = js([{"op": "series", "a": 0, "b": 1, "s": 0.1}, {"op": "series", "a": 1, "b": 0, "s": 0.25},
              {"op": "series", "a": 0.5, "b": 0.5, "s": 1}, {"op": "series", "a": 0, "b": 1, "s": 0}])
    assert got[0] == [round(0.1 * k, 10) for k in range(11)]
    assert got[1] == [1, 0.75, 0.5, 0.25, 0]
    assert got[2] == [0.5] and got[3] is None


def test_a_two_parameter_sweep_makes_every_combination_on_top_of_the_base_state():
    base = dict(J={"1": 1.0}, K1=0.177, H=0.5)
    got = js([{"op": "sweep", "base": base, "prefix": "disk",
               "rows": [dict(p="D", **{"from": 0, "to": 1, "step": 0.5}), dict(p="J2", **{"from": 0.1, "to": 0.2, "step": 0.1})]},
              {"op": "sweep", "base": base, "rows": [dict(p="H", **{"from": 0, "to": 10, "step": 0.001})], "limit": 1000},
              {"op": "sweep", "base": base, "rows": [dict(p="D", **{"from": 0, "to": 1, "step": 0.5}), dict(p="D", **{"from": 0, "to": 1, "step": 1})]}])
    states = got[0]["states"]
    assert [s["name"] for s in states] == ["disk D=0 J₂=0.1", "disk D=0 J₂=0.2", "disk D=0.5 J₂=0.1", "disk D=0.5 J₂=0.2",
                                           "disk D=1 J₂=0.1", "disk D=1 J₂=0.2"]
    assert states[3]["D"] == 0.5 and states[3]["J"] == {"1": 1.0, "2": 0.2} and states[3]["K1"] == 0.177
    assert "at most 1000" in got[1]["error"] and "only once" in got[2]["error"]


def test_the_package_writes_the_same_files_as_the_page():
    from Spin_Colour_Kernel import runfile
    sys.path.insert(0, str(ROOT / "notebooks"))
    import build_configurator as bc
    assert _same_build(runfile.TEMPLATE_PATH.read_text(), bc.template_json())   # the shipped template is current
    for cfg in (bc.DEFAULTS, FULL):
        full = runfile.config(cfg)
        nb, script = js([{"op": "files", "tpl": str(runfile.TEMPLATE_PATH), "cfg": full}])[0]
        assert runfile.notebook(full) == nb and runfile.script(full) == script


def test_the_package_refuses_an_unknown_setting():
    from Spin_Colour_Kernel import runfile
    with pytest.raises(ValueError, match="unknown setting 'boxx'"):
        runfile.config({"boxx": [4, 4]})


def test_the_reduced_box_check_agrees_with_the_package_on_every_bond_of_the_box():
    # random linear colourings, admissible or not, on boxes larger than the reduced window
    import random
    from Spin_Colour_Kernel import crystals as cr
    from Spin_Colour_Kernel.colouring import Colouring, verify_on_box
    rng = random.Random(5)
    reqs, want = [], []
    for _ in range(160):
        name = rng.choice(["square", "triangular", "honeycomb", "kagome", "sc", "bcc", "fcc"])
        shells = rng.choice([(1,), (1, 2)])
        c = cr.make(name, shells)
        m = rng.choice([(2,), (3,), (4,), (2, 2), (2, 4), (6,)])
        K = tuple(tuple(rng.randrange(mr) for _ in range(c.D)) for mr in m)
        o = tuple(tuple(rng.randrange(mr) for mr in m) for _ in range(c.b))
        box = tuple(rng.randint(1, 13) for _ in range(c.D))
        per = tuple(rng.random() < 0.5 for _ in range(c.D))
        want.append(verify_on_box(Colouring(m, K, o), c, box, per))
        reqs.append({"op": "verify", "m": m, "K": K, "o": o, "crystal": name, "shells": shells, "box": box,
                     "periodic": per})
    got = js(reqs)
    assert got == want and 0 < sum(want) < len(want)
