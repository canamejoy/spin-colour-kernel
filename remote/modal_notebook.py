"""Execute a notebook's code cells as one script on a Modal H100 (validation of the shipped notebooks).

    modal run remote/modal_notebook.py --notebook notebooks/nanodisk_simulation.ipynb \
        --override 'N_TEMPS = 4' --override 'SWEEPS = 3000'

Each --override replaces the line of the parameters cell that starts with the same name.
"""
import json
from pathlib import Path

import modal

image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", "numpy", "matplotlib"))
app = modal.App("sck-notebook", image=image)


@app.function(gpu="H100", timeout=3600)
def execute(script: str) -> dict:
    import os
    import subprocess
    import sys
    import tempfile
    d = tempfile.mkdtemp()
    Path(d, "nb.py").write_text(script)
    r = subprocess.run([sys.executable, "nb.py"], cwd=d, capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    res = Path(d, "results.json")
    return {"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-6000:],
            "results": res.read_text() if res.exists() else None}


@app.local_entrypoint()
def main(notebook: str, override: str = "", hours: float = 0.5):
    nb = json.loads(Path(notebook).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for ov in [o for o in override.split(";") if o.strip()]:
        name = ov.split("=")[0].strip()
        for i, c in enumerate(cells):
            lines = c.splitlines()
            hit = [k for k, ln in enumerate(lines) if ln.startswith(name + " ") or ln.startswith(name + "=")]
            if hit:
                end = hit[0]
                if lines[end].rstrip().endswith("[") or lines[end].count("[") > lines[end].count("]"):
                    depth = 0
                    for end in range(hit[0], len(lines)):
                        depth += lines[end].count("[") - lines[end].count("]")
                        if depth == 0:
                            break
                lines[hit[0]:end + 1] = [ov.strip()]
                cells[i] = "\n".join(lines)
                break
        else:
            raise SystemExit(f"override target {name} not found")
    script = "\n\n".join(cells)
    out = execute.with_options(timeout=int(hours * 3600)).remote(script)
    print(out["stdout"])
    print("--- stderr tail ---\n" + out["stderr"][-3000:])
    print("exit code", out["rc"])
    if out["results"]:
        dest = Path("results/h100/notebook-" + Path(notebook).stem + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(out["results"])
        print("results in", dest)
