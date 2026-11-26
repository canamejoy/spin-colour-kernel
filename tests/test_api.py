"""The pip interface: `import sck` is light, check/describe/write work, and run() runs and returns results."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
SMALL = dict(model="ising", crystal="square", box=[8, 8], periodic=[True, True],
             states=[{"name": "a", "J": {1: 1.0}}, {"name": "b", "J": {1: 1.0}, "H": 0.5}],
             T_max=3.0, T_min=2.0, T_step=1.0, thermalization=20, measurement=40, measure_every=5, replicas=4, seed=3)


def _py(code, cwd, **env):
    e = {**os.environ, "PYTHONPATH": SRC, **env}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, env=e, timeout=900)


def test_importing_sck_loads_neither_torch_nor_triton(tmp_path):
    r = _py("import sys, sck, Spin_Colour_Kernel as P; print(sorted(m for m in ('torch', 'triton') if m in sys.modules));"
            "print(P.run is sck.run, callable(sck.describe))", tmp_path)
    assert r.stdout.split() == ["[]", "True", "True"], r.stderr


def test_check_describe_and_write_through_the_short_names(tmp_path):
    import sck
    assert "ising" in sck.describe()["models"]
    assert sck.check(SMALL)["ok"] and not sck.check(SMALL, model="potts")["ok"]
    p = sck.write(tmp_path / "run.py", SMALL)
    assert p.suffix == ".py" and 'MODEL = "ising"' in p.read_text()
    with pytest.raises(ValueError, match="unknown model 'potts'"):
        sck.write(tmp_path / "bad.ipynb", SMALL, model="potts")


def test_run_refuses_a_bad_configuration_before_writing_anything(tmp_path):
    import sck
    with pytest.raises(ValueError, match="unknown model"):
        sck.run(SMALL, model="potts", output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_on_the_cpu_returns_results_and_a_figure(tmp_path):
    # the simulator is imported by check() before run() decides the device, as in a notebook session
    code = (f"import json, sck; cfg = json.loads({json.dumps(SMALL)!r}); assert sck.check(cfg)['ok']\n"
            "res = sck.run(cfg, output_dir='out', verbose=False)\n"
            "fig = res.plot('summary.png', show=False)\n"
            "print(json.dumps(dict(names=res.names, T=list(map(float, res.temperatures)), rows=len(res.table()),"
            " M=res.magnetization_per_site.tolist(), shape=list(res.configurations().shape), repr=repr(res))))")
    r = _py(code, tmp_path, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["names"] == ["a", "b"] and out["T"] == [3.0, 2.0] and out["rows"] == 4
    assert out["shape"] == [2, 4, 2, 64, 3] and (tmp_path / "summary.png").exists()
    assert all(0.0 <= m <= 1.0 for row in out["M"] for m in row) and out["M"][1][0] > out["M"][0][0]   # the field orders
    assert "2 state(s), 2 temperatures" in out["repr"]
    reloaded = _py("import sck; print(sck.load('out').names)", tmp_path)
    assert reloaded.stdout.strip() == "['a', 'b']", reloaded.stderr


def test_the_generator_keeps_floating_point_fusion_on_unless_the_user_chose(tmp_path):
    r = _py("import os; import Spin_Colour_Kernel.generator; print(os.environ['TRITON_DEFAULT_FP_FUSION'])", tmp_path)
    assert r.stdout.strip() == "1", r.stderr
    r = _py("import os; import Spin_Colour_Kernel.generator; print(os.environ['TRITON_DEFAULT_FP_FUSION'])", tmp_path,
            TRITON_DEFAULT_FP_FUSION="0")
    assert r.stdout.strip() == "0", r.stderr
