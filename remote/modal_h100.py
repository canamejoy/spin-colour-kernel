"""H100 runs on Modal, launched by the operator (experiments v03: E4, E6 H100 rows).

    modal run remote/modal_h100.py --exp E4,E4tri,E6             # full H100 set
    modal run remote/modal_h100.py --exp E4smoke,E6 --smoke      # minutes: checks the pipeline only
    modal run remote/modal_h100.py --exp E4 --only "Ising sc" --hours 3   # one E4 system, hard cost cap
    SCK_GPU=A100 modal run remote/modal_h100.py --exp E5conf,E6conf       # another GPU type (default H100)

`--hours` is a hard wall-clock limit on the container (cost cap: hours x the H100 rate); a run that hits it
is killed and returns whatever JSON it had already written.

Results land in the Modal volume `sck-results` under <run id>/ and are also returned to the local
`results/h100/<run id>/` directory when the call finishes.
"""

import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("SCK_GPU", "H100")          # Modal GPU type: H100, A100, A100-80GB, L4, ...
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    # BORIS (E6boris) links OpenMP, TBB, FFTW3, X11 and the system Python 3.10
    .apt_install("libomp-dev", "libtbb-dev", "libfftw3-dev", "libx11-dev", "python3-dev")
    # CUDA-METRO (E6metro); the add_python interpreter asks for clang++, so build pycuda with the image g++
    .run_commands("CC=gcc CXX=g++ CUDA_ROOT=/usr/local/cuda python -m pip install 'pycuda>=2024.1.2'")
    .pip_install("torch==2.5.1", "numpy", "scipy", "jax[cuda12]==0.4.35")
    # jax 0.4.35 reads nvidia.cuda_nvcc.__file__, which is None for that namespace package: use __path__
    .run_commands("sed -i 's/pathlib.Path(cuda_nvcc.__file__).parent/pathlib.Path(list(cuda_nvcc.__path__)[0])/' "
                  "/usr/local/lib/python3.11/site-packages/jax/_src/lib/__init__.py")
    # Nsight Compute for E5prof (best effort: the image still builds if the package is missing)
    .run_commands("apt-get update && V=$(apt-cache search --names-only '^nsight-compute-20' | awk '{print $1}' "
                  "| sort -V | tail -1) && (apt-get install -y --no-install-recommends $V || true) "
                  "&& rm -rf /var/lib/apt/lists/*")
    # SCK_GPU travels into the container so the function is defined there with the same GPU type
    .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "CUDA_HOME": "/usr/local/cuda", "SCK_GPU": GPU})
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_dir(ROOT / "third_party", "/root/third_party")
)
app = modal.App("spin-colour-kernel", image=image)
vol = modal.Volume.from_name("sck-results", create_if_missing=True)


@app.function(gpu=GPU, timeout=24 * 3600, volumes={"/results": vol})
def run(exp: str, smoke: bool, run_id: str, only: str = "", limit_s: int = 0, meas: int = 20000) -> dict:
    import os
    import subprocess
    import sys

    out = f"/results/{run_id}"
    env = dict(os.environ, PYTHONPATH="/root/src", SCK_CACHE="/tmp/sck_cache")
    subprocess.run(["nvidia-smi"])
    cmd = [sys.executable, "-m", "Spin_Colour_Kernel.run", "--exp", exp, "--out", out, "--device", "cuda:0"]
    cmd += (["--smoke"] if smoke else []) + (["--only", only] if only else []) + ["--meas", str(meas)]
    try:
        subprocess.run(cmd, env=env, check=False, timeout=limit_s or None)
    except subprocess.TimeoutExpired:
        print("time limit reached", flush=True)
    vol.commit()
    files = {}
    for p in Path(out).iterdir():      # JSON summaries and the raw E4 series (.npz)
        if p.suffix in (".json", ".npz", ".ncu-rep", ".ncu-repz"):
            files[p.name] = p.read_bytes()
    return files


@app.local_entrypoint()
def main(exp: str = "E4,E4tri,E6", smoke: bool = False, only: str = "", hours: float = 24.0,
         meas: int = 20000):
    import time

    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if smoke else "")
    if only:
        run_id += "-" + only.replace(" ", "_").replace(",", "-")
    if meas != 20000:
        run_id += f"-meas{meas}"
    limit = int(hours * 3600)
    # the container is killed at limit + 10 min whatever happens; the subprocess stops at the limit itself
    files = run.with_options(timeout=limit + 600).remote(exp, smoke, run_id, only, limit, meas)
    dest = ROOT / "results" / GPU.lower().split("-")[0] / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (dest / name).write_bytes(text)
    print("results in", dest)
