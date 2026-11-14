"""The notebook builders embed every package module their modules import."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "Spin_Colour_Kernel"


def _builder(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT / "notebooks"))
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["build_nanodisk_notebook", "build_simulation_notebook"])
def test_every_relative_import_of_an_embedded_module_is_embedded(name):
    modules = set(_builder(name).MODULES)
    needed = set()
    for m in modules:
        src = (PKG / m).read_text()
        needed |= {f"{x}.py" for x in re.findall(r"^from \.(\w+) import", src, re.M)}
        needed |= {f"{x}.py" for x in re.findall(r"^from \. import (\w+)", src, re.M)}
    assert needed <= modules, f"{name} misses {sorted(needed - modules)}"


def test_the_simulation_notebook_has_one_parameters_cell_with_every_setting():
    nb = _builder("build_simulation_notebook").build()
    src = "".join(nb["cells"][2]["source"])
    for key in ("MODEL", "STATES", "UNITS", "CRYSTAL", "BOX", "PERIODIC", "COORDS", "SHAPE", "T_MAX", "T_MIN",
                "T_STEP", "THERMALIZATION", "MEASUREMENT", "MEASURE_EVERY", "REPLICAS", "CONFIG_EVERY", "GPUS"):
        assert re.search(rf"^{key}\s*=", src, re.M), key
