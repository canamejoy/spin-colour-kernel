"""Why do u = 4 grids of the XLA baseline run faster? Nsight Compute on the E5 baseline step.

For each grid the driver runs the inner process under `ncu`: the step is compiled with XLA's dump enabled,
warmed up, and one call is profiled between cudaProfilerStart and cudaProfilerStop. Per kernel it keeps
the time, the launch shape, DRAM traffic, executed instructions, global-load instructions, L1 requests,
sectors and bytes of global loads, occupancy and the warp-stall breakdown; from the dumped PTX it keeps
the width histogram of global loads and stores of every kernel. Together they separate the candidate
mechanisms: fewer threads with more work each (launch shape), vector loads (bytes per thread per load
instruction, `ld.global.v4` in PTX), coalescing (sectors per request) and instruction count.

    python -m Spin_Colour_Kernel.e5_profile --out results/ [--grids heis:79,80,81 ising:79,80,81] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KEEP = ("gpu__time_duration.sum", "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
        "dram__bytes_read.sum", "dram__bytes_write.sum", "smsp__inst_executed.sum",
        "smsp__sass_inst_executed_op_global_ld.sum", "smsp__sass_inst_executed_op_global_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct", "launch__waves_per_multiprocessor",
        "sm__warps_active.avg.per_cycle_active", "smsp__thread_inst_executed.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "launch__occupancy_limit_registers", "sm__maximum_warps_per_active_cycle_pct")
STALL = re.compile(r"^smsp__average_warps?_(?:latency_)?issue_stalled_(\w+?)(?:_per_issue_active)?\.ratio$")
GRID = {"heis": (21, 3, 1), "ising": (63, 1, 1)}      # (B, n, b) of the E5 L1 x 31^2 sweeps


def _num(v: str):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return v


def parse_raw_csv(text: str) -> list[dict]:
    """`ncu --import --csv --page raw`: a header row, a units row, then one row per kernel launch."""
    rows = list(csv.reader(io.StringIO(text)))
    head = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if head is None:
        return []
    names = rows[head]
    units = dict(zip(names, rows[head + 1])) if len(rows) > head + 1 else {}
    out = []
    for r in rows[head + 2:]:
        if len(r) != len(names):
            continue
        d = dict(zip(names, r))
        k = {"kernel": d.get("Kernel Name", ""), "id": d.get("ID", "")}
        for m in KEEP:
            if m in d:
                k[m] = _num(d[m])
        k["stalls"] = {STALL.match(m).group(1): _num(v) for m, v in d.items() if STALL.match(m)}
        k["units"] = {m: units.get(m, "") for m in KEEP if m in d}
        out.append(k)
    return out


def derived(k: dict) -> dict:
    """Bytes per thread per global-load instruction (4 scalar, 8 v2, 16 v4) and sectors per request."""
    rq = k.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") or 0
    se = k.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") or 0
    g, bl = k.get("launch__grid_size") or 0, k.get("launch__block_size") or 0
    # one request per warp-level load instruction, 32-byte sectors, 32 threads: a fully coalesced load moves
    # sectors/request x 32 B / 32 threads = sectors/request bytes per thread (4 scalar f32, 16 for float4)
    spr = se / rq if rq else None
    return {"threads": g * bl if isinstance(g, float) and isinstance(bl, float) else None,
            "sectors_per_load_request": spr, "bytes_per_thread_per_load": spr}


def ptx_histograms(dump: Path) -> list[dict]:
    from Spin_Colour_Kernel.width_law import ptx_widths
    out = []
    for f in glob.glob(str(dump / "**" / "*.ptx"), recursive=True):
        for k in ptx_widths(Path(f).read_text(errors="ignore")):
            if k["hist"]:
                out.append({"file": os.path.basename(f)[-80:], **k})
    return out


def inner(spec: str, dump: Path, out_json: Path):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={dump}"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from Spin_Colour_Kernel.width_law_b import make_baseline_step
    kind, L1 = spec.split(":")
    B, n, b = GRID[kind]
    box = (int(L1), 31, 31)
    f = make_baseline_step(n, b, box, int(L1))
    S = jnp.ones((B, n, b, box[2], box[1], box[0]), jnp.float32) / np.sqrt(n)
    key = jax.random.PRNGKey(1)
    for _ in range(3):
        S, key = f(S, key, 0.5)
    S.block_until_ready()
    rt = None
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            pass
    if rt is None:      # JAX's pip CUDA runtime
        import nvidia.cuda_runtime as cr_
        rt = ctypes.CDLL(str(Path(list(cr_.__path__)[0]) / "lib" / "libcudart.so.12"))
    rt.cudaProfilerStart()
    S, key = f(S, key, 0.5)
    S.block_until_ready()
    rt.cudaProfilerStop()
    out_json.write_text(json.dumps({"spec": spec, "B": B, "n": n, "b": b, "box": list(box),
                                    "elements": B * n * b * box[0] * box[1] * box[2], "ptx": ptx_histograms(dump)}))


# generated-kernel targets (route A): name -> (system builder, batch, packed)
GEN = {"ising-square-1024-int8": ("ising", "square", 1024, 16, False),
       "ising-square-1024-packed": ("ising", "square", 1024, 32, True),
       "heis-sc-32": ("heisenberg", "sc", 32, 128, None),
       "ising-square-512-bitwise": ("ising", "square", 512, 64, "bitwise"),
       "ising-square-512-independent": ("ising", "square", 512, 64, "independent"),
       "ising-square-4096-independent": ("ising", "square", 4096, 64, "independent"),
       # large-lattice diagnostic (v14 CUDA-METRO pairing): batch 1, 2D triangular Heisenberg
       "heis-triangular-1024-b1": ("heisenberg", "triangular", 1024, 1, None),
       "heis-triangular-2048-b1": ("heisenberg", "triangular", 2048, 1, None),
       "heis-triangular-4096-b1": ("heisenberg", "triangular", 4096, 1, None)}


def inner_generated(spec: str, out_json: Path):
    """One profiled coloured sweep of the generated Triton kernel (route A of close-reviewer-gaps)."""
    import torch
    from Spin_Colour_Kernel import system as sy
    from Spin_Colour_Kernel.sampler import Sampler
    kind, crystal, L, B, packed = GEN[spec]
    s = sy.ising(crystal, L, T_ref=sy.TC["Ising square"]) if kind == "ising" else sy.heisenberg(crystal, L)
    kw = {"bitwise": True, "words_per_program": 8, "independent": packed == "independent"} \
        if packed in ("bitwise", "independent") else {"packed": packed}
    A = Sampler(s, B, "cuda:0", 1, **kw)
    A.set_temperature(s.T_ref or 1.0)
    S = A.init_spins(seed=1)
    A.sweep(S, 3)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    A.sweep(S, 1)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    out_json.write_text(json.dumps({"spec": spec, "B": B, "packed": A.packed, "sites": s.n_active,
                                    "cblock": getattr(A, "cblock", None), "rows_per": getattr(A, "rows_per", None),
                                    "bx": getattr(A, "bx", None), "stride": getattr(A, "stride", None),
                                    "cnw": getattr(A, "cnw", None),
                                    "colours": s.colouring.n_colours, "kernel_source_lines": A.src.count("\n")}))


def find_ncu() -> str | None:
    import re
    hits = glob.glob("/opt/nvidia/nsight-compute/*/ncu") + glob.glob("/usr/local/cuda*/bin/ncu")
    ver = lambda h: [int(x) for x in re.findall(r"\d+", h)]          # newest release first
    return max(hits, key=ver) if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--grids", nargs="*", default=[f"heis:{L}" for L in (75, 76, 77, 79, 80, 81, 83, 84, 85)]
                    + ["ising:79", "ising:80", "ising:81"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--generated", action="store_true", help="profile the generated kernels (GEN) instead of XLA")
    ap.add_argument("--inner", default=None)
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.inner:
        if a.inner in GEN:
            inner_generated(a.inner, a.json)
        else:
            inner(a.inner, a.dump, a.json)
        return
    ncu = find_ncu()
    a.out.mkdir(parents=True, exist_ok=True)
    grids = (a.grids if a.grids and a.grids[0] in GEN else list(GEN)) if a.generated else a.grids
    grids = grids[:2] if a.quick else grids
    result = {"ncu": ncu, "grids": []}
    if ncu is None:
        result["error"] = "ncu not found"
    for spec in ([] if ncu is None else grids):
        work = Path(tempfile.mkdtemp(prefix="e5prof_"))
        rep = work / "prof"
        cmd = [ncu, "--profile-from-start", "off", "--target-processes", "all", "--set", "full", "--clock-control", "none",
               "--export", str(rep), "--force-overwrite",
               sys.executable, "-m", "Spin_Colour_Kernel.e5_profile", "--inner", spec,
               "--dump", str(work / "dump"), "--json", str(work / "inner.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        entry = {"spec": spec, "ncu_exit": r.returncode, "ncu_tail": (r.stdout + r.stderr)[-3000:]}
        if (work / "inner.json").exists():
            entry.update(json.loads((work / "inner.json").read_text()))
        reports = glob.glob(str(rep) + ".ncu-rep*")          # .ncu-rep, or .ncu-repz from Nsight 2026
        if reports:
            raw = subprocess.run([ncu, "--import", reports[0], "--csv", "--page", "raw"],
                                 capture_output=True, text=True).stdout
            kernels = parse_raw_csv(raw)
            hdr = next((r for r in csv.reader(io.StringIO(raw)) if "Kernel Name" in r), [])
            entry["stall_metric_names"] = [m for m in hdr if "stall" in m][:120]
            for k in kernels:
                k.update(derived(k))
            entry["kernels"] = kernels
            entry["step_time_ns"] = sum(k.get("gpu__time_duration.sum") or 0 for k in kernels
                                        if isinstance(k.get("gpu__time_duration.sum"), float))
            shutil.copy(reports[0], a.out / f"E5prof-{spec.replace(':', '-')}{Path(reports[0]).suffix}")
        result["grids"].append(entry)
        print(f"E5prof {spec}: ncu exit {r.returncode}, kernels {len(entry.get('kernels', []))}, "
              f"step {entry.get('step_time_ns', 0) / 1e3:.1f} us", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    (a.out / ("Kprof.json" if a.generated else "E5prof.json")).write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
