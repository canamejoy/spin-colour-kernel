"""Run tools of the MCP server: launch a checked run where the user has access, follow it and read its results.

Backends:
    local    this machine: the .py run starts detached; its log and results stay in the job folder
    kaggle   a private GPU notebook pushed with the kaggle CLI; results come back with `kaggle kernels output`
    modal    the .py run on a Modal GPU through the modal CLI; results go to the Modal Volume "sck-runs"
    runpod   a GPU pod created through the RunPod REST API v2 (key in RUNPOD_API_KEY); the pod serves its
             results behind a random per-job token and is terminated once they are downloaded
    colab    no submission: the notebook is written for the user (or Google's Colab MCP server) to open

Every job is a folder under SCK_JOBS_DIR (default ./sck_jobs) holding job.json, the run file, the log and the
results. A remote launch spends the user's money or quota: without confirm=true, run only returns the plan.

Credentials are detected by presence only (CLI on PATH, credential file present, environment variable name
set); no credential value is ever read into a tool result, a job record or a log. The only value read is the
Kaggle username, needed to name the kernel.

    python -m Spin_Colour_Kernel.jobs pod     # the program a RunPod pod runs (see _pod_main)
"""

from __future__ import annotations

import base64
import hmac
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import agent, mcp, runfile
from .mcp import ToolError

BACKENDS = ("local", "kaggle", "modal", "runpod", "colab")
REMOTE = ("kaggle", "modal", "runpod")
SECRET_ENV = ("KAGGLE_KEY", "KAGGLE_API_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "RUNPOD_API_KEY")
RUNPOD_API = "https://api.runpod.io/v2"
RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REPO = "https://github.com/canamejoy/spin-colour-kernel"
MODAL_VOLUME = "sck-runs"
DEFAULTS = {
    "local": dict(max_hours=12.0),
    "kaggle": dict(accelerator="NvidiaTeslaT4", max_hours=9.0),
    "modal": dict(gpu="L4", max_hours=2.0),
    "runpod": dict(gpu="NVIDIA L4", image=RUNPOD_IMAGE, ref="v1.5.1", max_hours=2.0),
    "colab": {},
}
_PROCS: dict[str, subprocess.Popen] = {}           # children started by this server (polled to avoid zombies)
_ID = re.compile(r"^(local|kaggle|modal|runpod|colab)-\d{8}-\d{6}-[0-9a-f]{6}$")


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────

def jobs_dir() -> Path:
    return Path(os.environ.get("SCK_JOBS_DIR") or "sck_jobs").expanduser().resolve()


def _scrub(text: str) -> str:
    """Remove any credential value that a tool might echo (defensive: none should ever reach here)."""
    for k in SECRET_ENV:
        v = os.environ.get(k)
        if v and len(v) >= 4:
            text = text.replace(v, "***")
    return text


def _sh(cmd, timeout=600, cwd=None) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except FileNotFoundError:
        raise ToolError(f"{cmd[0]} is not installed (not on PATH)")
    except subprocess.TimeoutExpired:
        raise ToolError(f"{' '.join(cmd[:3])} did not answer within {timeout} s")
    r.stdout, r.stderr = _scrub(r.stdout or ""), _scrub(r.stderr or "")
    return r


def _tail(path: Path, n=4000) -> str:
    try:
        return _scrub(path.read_text(errors="replace")[-n:])
    except OSError:
        return ""


def _load(job) -> tuple[dict, Path]:
    if not isinstance(job, str) or not _ID.match(job):
        raise ToolError(f"unknown job {job!r}; job ids look like local-20261123-101500-a1b2c3")
    d = jobs_dir() / job
    try:
        return json.loads((d / "job.json").read_text()), d
    except FileNotFoundError:
        raise ToolError(f"no job {job} in {jobs_dir()}")


def _save(rec: dict, d: Path) -> None:
    (d / "job.json").write_text(json.dumps(rec, indent=1))


def _view(rec: dict) -> dict:
    """The record as tools show it: the per-job download token stays private."""
    return {k: v for k, v in rec.items() if k != "token"}


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _hours(v, cap=None) -> float:
    h = float(v)
    if not math.isfinite(h) or h <= 0:
        raise ToolError("max_hours must be a positive number")
    return min(h, cap) if cap else h


# ── detection (S4: presence only) ────────────────────────────────────────────────────────────────

def _tool(cli, module=None) -> bool:
    return bool(shutil.which(cli)) or bool(module and importlib.util.find_spec(module))


def _env_set(*names) -> bool:
    return all(os.environ.get(n) for n in names)


def kaggle_user() -> str | None:
    """The Kaggle username (never the key): KAGGLE_USERNAME, else the "username" field of kaggle.json."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for f in _kaggle_json():
        try:
            u = json.loads(f.read_text()).get("username")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(u, str) and u:
            return u
    return None


def _kaggle_json() -> list[Path]:
    dirs = [Path(os.environ["KAGGLE_CONFIG_DIR"])] if os.environ.get("KAGGLE_CONFIG_DIR") else []
    dirs += [Path.home() / ".kaggle", Path.home() / ".config" / "kaggle"]
    return [d / "kaggle.json" for d in dirs if (d / "kaggle.json").is_file()]


def detect() -> dict:
    gpu = bool(shutil.which("nvidia-smi"))
    torch = bool(importlib.util.find_spec("torch"))
    out = {"local": dict(
        available=torch, gpu=gpu,
        reason=("torch installed" if torch else "torch is not installed")
        + ("; an NVIDIA driver is present" if gpu else "; no nvidia-smi: the run would use the CPU (slow)"),
        what_to_install_or_configure="" if torch else "pip install torch numpy matplotlib")}

    cli = _tool("kaggle", "kaggle")
    creds = bool(_kaggle_json()) or _env_set("KAGGLE_USERNAME", "KAGGLE_KEY") or _env_set("KAGGLE_API_TOKEN") \
        or (Path.home() / ".kaggle" / "access_token").is_file()
    user = kaggle_user()
    miss = ([] if cli else ["pip install kaggle"]) + ([] if creds else [
        "an API token from kaggle.com/settings in ~/.kaggle/kaggle.json (or KAGGLE_USERNAME and KAGGLE_KEY)"]) \
        + ([] if user or not creds else ["KAGGLE_USERNAME (the kernel is named after the user)"])
    out["kaggle"] = dict(available=cli and creds and bool(user), user=user,
                         reason="kaggle CLI and credentials found" if not miss else "missing: " + "; ".join(miss),
                         what_to_install_or_configure="; ".join(miss))

    cli = _tool("modal")
    toml = Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml").expanduser()
    creds = toml.is_file() or _env_set("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    miss = ([] if cli else ["pip install modal (the modal CLI must be on PATH)"]) + (
        [] if creds else ["modal token new (writes ~/.modal.toml) or MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"])
    out["modal"] = dict(available=cli and creds,
                        reason="modal CLI and credentials found" if not miss else "missing: " + "; ".join(miss),
                        what_to_install_or_configure="; ".join(miss))

    key = _env_set("RUNPOD_API_KEY")
    out["runpod"] = dict(available=key, reason="RUNPOD_API_KEY is set" if key else "RUNPOD_API_KEY is not set",
                         what_to_install_or_configure="" if key else
                         "an API key from runpod.io (Settings, API Keys) in the RUNPOD_API_KEY environment variable "
                         "of the MCP server")
    out["colab"] = dict(available=True, reason="no submission API: the notebook is written for the user to open "
                        "in Colab, or for Google's Colab MCP server (googlecolab/colab-mcp) if the agent has it",
                        what_to_install_or_configure="")
    return out


def _backends(args):
    return detect()


# ── run ──────────────────────────────────────────────────────────────────────────────────────────

def _options(backend, given) -> dict:
    opts = dict(DEFAULTS[backend])
    for k, v in (given or {}).items():
        if k not in opts:
            raise ToolError(f"unknown option {k!r} for {backend}; options: {', '.join(opts) or 'none'}")
        opts[k] = v
    if "max_hours" in opts:
        opts["max_hours"] = _hours(opts["max_hours"], 24.0 if backend == "modal" else None)
    if backend == "runpod":
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", str(opts["ref"])):
            raise ToolError("ref must be a git tag, branch or commit")
    return opts


def _plan(backend, r, opts, user_cfg, results) -> dict:
    b = r["budget"]
    gpu = {"local": "this machine's GPU(s), or its CPU", "kaggle": opts.get("accelerator"),
           "modal": opts.get("gpu"), "runpod": opts.get("gpu"), "colab": "the Colab runtime the user picks"}[backend]
    limit = {"local": f"the run is stopped after {opts.get('max_hours')} h",
             "kaggle": f"Kaggle stops the session after {opts.get('max_hours')} h (and at its weekly GPU quota)",
             "modal": f"the Modal function timeout is {opts.get('max_hours')} h, then it is stopped",
             "runpod": f"the run is stopped after {opts.get('max_hours')} h, but the pod keeps billing at its GPU "
                       "rate until it is terminated: call results (downloads, then terminates) or cancel as soon as "
                       "status says finished",
             "colab": "Colab's own session limits"}[backend]
    plan = dict(backend=backend, gpu=gpu,
                cost_drivers=dict(chains=b["chains"], spin_updates=b["spin_updates"],
                                  temperatures=r["temperatures"]["count"], output_gb=b["output_gb"]),
                time_limit=limit, results_will_land=results, options=opts, warnings=r.get("warnings", []))
    if backend == "runpod":
        plan["cost_per_hour"] = "set by RunPod for the GPU type; status reports the pod's cost per hour once created"
        plan["billing"] = ("the pod bills from creation until results or cancel terminates it, also after the run "
                           "ends; the run itself is capped at max_hours")
    if backend in REMOTE or backend == "local":
        plan["next_call"] = {"tool": "run", "arguments": {"config": user_cfg, "backend": backend, "confirm": True,
                                                          "options": opts}}
        plan["note"] = ("nothing was submitted: this spends the user's " +
                        ("machine time" if backend == "local" else "money or quota") +
                        "; call run again with confirm: true only if the user asked for it")
    return plan


def _run(args):
    backend = args.get("backend") or "local"
    if backend not in BACKENDS:
        raise ToolError(f"backend must be one of {', '.join(BACKENDS)}")
    user_cfg = args.get("config") or {}
    r = agent.check(user_cfg)
    if not r["ok"]:
        raise ToolError("the configuration has errors, nothing was launched: " + "; ".join(r["errors"]))
    cfg = r["config"]
    outdir = str(cfg.get("output_dir") or "simulation_output")
    if Path(outdir).is_absolute() or ".." in Path(outdir).parts:
        raise ToolError("output_dir must be a relative folder without '..' for a job")
    opts = _options(backend, args.get("options"))
    where = {"local": f"{jobs_dir()}/<job>/{outdir}", "colab": "wherever the user saves them in Colab",
             "modal": f"Modal Volume {MODAL_VOLUME}:/<job>/{outdir}, downloaded by results to {jobs_dir()}/<job>/results"
             }.get(backend, f"{jobs_dir()}/<job>/results (downloaded by results)")
    if backend != "colab" and args.get("confirm") is not True:   # only a real true submits
        return {"submitted": False, "plan": _plan(backend, r, opts, user_cfg, where)}
    avail = detect()[backend]
    if not avail["available"]:
        raise ToolError(f"{backend} is not available: {avail['reason']}")
    job = f"{backend}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    d = jobs_dir() / job
    d.mkdir(parents=True)
    rec = dict(id=job, backend=backend, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), options=opts,
               output_dir=outdir, budget=r["budget"], state="submitting")
    _save(rec, d)
    try:
        globals()[f"_submit_{backend}"](rec, d, cfg)
    except Exception:
        rec["state"] = "failed to submit"
        _save(rec, d)
        raise
    _save(rec, d)
    return {"submitted": backend != "colab", "job": _view(rec), "warnings": r.get("warnings", []),
            "next": rec.get("instructions") or f"call status with job {job!r}, then results when it has finished"}


def _submit_local(rec, d, cfg):
    runfile.write(cfg, d / "run.py", "py")
    wrap = ("import pathlib, subprocess, sys\n"
            "try:\n    rc = subprocess.run([sys.executable, '-u', 'run.py'], timeout=float(sys.argv[1])).returncode\n"
            "except subprocess.TimeoutExpired:\n    rc = 'timeout'\n"
            "pathlib.Path('EXIT').write_text(str(rc))\n")
    with open(d / "run.log", "wb") as log:
        p = subprocess.Popen([sys.executable, "-c", wrap, str(rec["options"]["max_hours"] * 3600)], cwd=d,
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                             env=dict(os.environ, MPLBACKEND="Agg"))
    _PROCS[rec["id"]] = p
    rec.update(pid=p.pid, state="running", results_dir=str(d / rec["output_dir"]), log=str(d / "run.log"))


def _submit_colab(rec, d, cfg):
    path = runfile.write(cfg, d / "run.ipynb", "ipynb")
    rec.update(state="handed off", notebook=str(path), results_dir=str(d / "results"), instructions=(
        f"Colab has no submission API here. Open {path} in Colab (colab.research.google.com, File, Upload "
        "notebook), pick a GPU runtime and press Run all; or, if the agent has Google's Colab MCP server "
        "(googlecolab/colab-mcp), hand this notebook to it. To read the results here, download the folder "
        f"{rec['output_dir']} from Colab into {d / 'results'} and call results."))


def _kaggle_slug(rec):
    return f"sck-{rec['id']}"


def _submit_kaggle(rec, d, cfg):
    user = kaggle_user()
    k = d / "kernel"
    k.mkdir()
    runfile.write(cfg, k / "run.ipynb", "ipynb")
    slug = _kaggle_slug(rec)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (k / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    cmd = ["kaggle", "kernels", "push", "-p", str(k)]
    flags = _sh(["kaggle", "kernels", "push", "--help"], timeout=60).stdout
    if "--accelerator" in flags:
        cmd += ["--accelerator", str(rec["options"]["accelerator"])]
    if "--timeout" in flags:
        cmd += ["-t", str(int(rec["options"]["max_hours"] * 3600))]
    r = _sh(cmd)
    (d / "push.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise ToolError(f"kaggle kernels push failed: {(r.stderr or r.stdout)[-1500:]}")
    rec.update(kernel=meta["id"], state="submitted", results_dir=str(d / "results"),
               url=f"https://www.kaggle.com/code/{meta['id']}", push_output=r.stdout[-800:])


MODAL_APP = '''"""Run one SCK job on a Modal GPU (written by Spin_Colour_Kernel.jobs)."""
import os, pathlib, subprocess, sys
import modal

JOB, GPU, TIMEOUT = {job!r}, {gpu!r}, {timeout}
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", "numpy", "matplotlib"))
vol = modal.Volume.from_name({volume!r}, create_if_missing=True)
app = modal.App("sck-" + JOB, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={{"/runs": vol}})
def run(script: str) -> int:
    d = pathlib.Path("/runs") / JOB
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(script)
    with open(d / "run.log", "w") as log:
        rc = subprocess.call([sys.executable, "-u", "run.py"], cwd=d, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, MPLBACKEND="Agg"))
    (d / "EXIT").write_text(str(rc))
    vol.commit()
    return rc


@app.local_entrypoint()
def main():
    print("SCK_EXIT", run.remote(pathlib.Path(__file__).with_name("run.py").read_text()))
'''


def _submit_modal(rec, d, cfg):
    o = rec["options"]
    runfile.write(cfg, d / "run.py", "py")
    (d / "modal_app.py").write_text(MODAL_APP.format(job=rec["id"], gpu=str(o["gpu"]), volume=MODAL_VOLUME,
                                                     timeout=int(o["max_hours"] * 3600)))
    with open(d / "modal.log", "wb") as log:     # -d: the app keeps running if this process dies
        p = subprocess.Popen(["modal", "run", "-d", "modal_app.py"], cwd=d, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    _PROCS[rec["id"]] = p
    rec.update(pid=p.pid, state="submitted", results_dir=str(d / "results"), log=str(d / "modal.log"),
               volume=f"{MODAL_VOLUME}:/{rec['id']}")


def _runpod(method, path, body=None, timeout=60):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise ToolError("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(RUNPOD_API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode() or "{}"
    except urllib.error.HTTPError as e:
        raise ToolError(f"RunPod {method} {path}: HTTP {e.code} {_scrub(e.read().decode(errors='replace'))[:800]}")
    except urllib.error.URLError as e:
        raise ToolError(f"RunPod {method} {path}: {e.reason}")
    return json.loads(raw)


def runpod_body(rec, cfg, token) -> dict:
    o = rec["options"]
    disk = int(math.ceil(float(rec["budget"]["output_gb"]))) + 30
    boot = (f"pip install -q 'git+{REPO}@{o['ref']}' && "
            "mkdir -p /workspace/sck && cd /workspace/sck && python -m Spin_Colour_Kernel.jobs pod")
    return {"name": f"sck-{rec['id']}", "image": str(o["image"]), "gpu": {"id": str(o["gpu"]), "count": 1},
            "disk": disk, "ports": ["8000/http"],
            "env": {"SCK_CONFIG": base64.b64encode(json.dumps(cfg).encode()).decode(), "SCK_TOKEN": token,
                    "SCK_MAX_HOURS": str(o["max_hours"])},
            "cmd": ["bash", "-lc", boot]}


def _submit_runpod(rec, d, cfg):
    token = secrets.token_urlsafe(24)
    pod = _runpod("POST", "/pods", runpod_body(rec, cfg, token))
    if not pod.get("id"):
        raise ToolError(f"RunPod returned no pod id: {_scrub(json.dumps(pod))[:500]}")
    rec.update(pod=pod["id"], token=token, state="submitted", results_dir=str(d / "results"),
               proxy=f"https://{pod['id']}-8000.proxy.runpod.net")


# ── status / results / cancel ────────────────────────────────────────────────────────────────────

def _local_state(rec, d, marker: Path) -> str:
    if rec.get("state") == "cancelled":
        return "cancelled"
    p = _PROCS.get(rec["id"])
    if p is not None:
        p.poll()
    if marker.exists():
        code = marker.read_text().strip()
        return "finished" if code == "0" else f"failed (exit {code})"
    if (p is not None and p.returncode is None) or (p is None and _alive(rec.get("pid"))):
        return "running"
    return "failed (stopped without an exit marker; read the log)"


def _rp_url(rec, name):
    return f"{rec['proxy']}/{rec['token']}/{name}"


def _http_get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return resp.read()


def _status_of(rec, d) -> dict:
    b, out = rec["backend"], {}
    if b == "local":
        rec["state"] = _local_state(rec, d, d / "EXIT")
        out["log_tail"] = _tail(d / "run.log", 1500)
    elif b == "colab":
        out["note"] = ("Colab runs are not tracked by this server: follow the notebook in Colab (or through the "
                       "Colab MCP server); put the downloaded output folder in results_dir and call results")
    elif b == "kaggle":
        r = _sh(["kaggle", "kernels", "status", rec["kernel"]], timeout=120)
        text = (r.stdout + r.stderr).strip()
        low = text.lower()
        rec["state"] = ("finished" if "complete" in low else "failed" if "error" in low else
                        "cancelled" if "cancel" in low else "queued" if "queued" in low else
                        "running" if "running" in low else rec["state"])
        out["kaggle"] = text[-500:]
    elif b == "modal":
        r = _sh(["modal", "volume", "get", MODAL_VOLUME, f"{rec['id']}/EXIT", "-"], timeout=120)
        # the CLI appends its own progress line ("✓ Finished downloading files to local!") to the file's text
        m = re.match(r"\s*(-?\d+)", r.stdout or "") if r.returncode == 0 else None
        code = m.group(1) if m else ""
        if code:
            rec["state"] = "finished" if code == "0" else f"failed (exit {code})"
        else:
            p = _PROCS.get(rec["id"])
            running = (p.poll() is None) if p is not None else _alive(rec.get("pid"))
            rec["state"] = "running" if running else "stopped without an exit marker (timeout or error; read the log)"
        out["log_tail"] = _tail(d / "modal.log", 1500)
    elif b == "runpod":
        pod = _runpod("GET", f"/pods/{rec['pod']}")
        out["pod_status"] = pod.get("status") or pod.get("desiredStatus")
        cost = next((pod[k] for k in ("costPerHr", "costPerHour", "cost") if k in pod), None)
        if cost is not None:
            rec["cost_per_hour_usd"] = cost
        if out["pod_status"] in ("TERMINATED", "ERROR") and rec["state"] not in ("finished", "fetched"):
            rec["state"] = f"pod {out['pod_status'].lower()}"
        elif out["pod_status"] == "RUNNING":
            try:
                code = _http_get(_rp_url(rec, "DONE")).decode().strip()
                rec["state"] = "finished" if code == "0" else f"failed (exit {code})"
            except (urllib.error.URLError, OSError):
                rec["state"] = "running"
        elif out["pod_status"] == "EXITED":
            rec["state"] = "pod exited"
        else:
            rec["state"] = "starting"
    _save(rec, d)
    return {"job": _view(rec), **out}


def _status(args):
    rec, d = _load(args.get("job"))
    return _status_of(rec, d)


def _fetch(rec, d, keep):
    b, res = rec["backend"], d / "results"
    if b == "kaggle":
        res.mkdir(exist_ok=True)
        r = _sh(["kaggle", "kernels", "output", rec["kernel"], "-p", str(res), "-o"], timeout=3600)
        if r.returncode != 0:
            raise ToolError(f"kaggle kernels output failed: {(r.stderr or r.stdout)[-1000:]}")
    elif b == "modal":
        res.mkdir(exist_ok=True)
        r = _sh(["modal", "volume", "get", "--force", MODAL_VOLUME, f"{rec['id']}/{rec['output_dir']}", str(res)],
                timeout=3600)
        if r.returncode != 0:
            raise ToolError(f"modal volume get failed: {(r.stderr or r.stdout)[-1000:]}")
    elif b == "runpod":
        try:
            _fetch_runpod(rec, res)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise ToolError(f"downloading from pod {rec['pod']} failed ({e}); the pod is still running and billing: "
                            "call results again, or cancel to terminate it") from None
        if not keep:
            _runpod("DELETE", f"/pods/{rec['pod']}")
            rec["pod_terminated"] = True
    rec["state"] = "fetched"


def _fetch_runpod(rec, res):
    names = json.loads(_http_get(_rp_url(rec, ""), timeout=90).decode())["files"]
    for name in names:
        dest = (res / name).resolve()
        if res.resolve() not in dest.parents:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_rp_url(rec, urllib.request.quote(name)), timeout=95) as resp, \
                open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)


def summarize(folder: Path) -> dict:
    """Files and sizes, and per state |M| and energy per site against T, read from results.npz only."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []
    out = dict(folder=str(folder), files=[{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size}
                                          for p in files[:200]])
    npz = next((p for p in files if p.name == "results.npz"), None)
    png = next((p for p in files if p.name == "summary.png"), None)
    out["summary_png"] = str(png) if png else None
    if npz is None:
        out["note"] = "no results.npz yet"
        return out
    import numpy as np                              # only here: the server starts without it
    with np.load(npz) as z:
        T = z["temperatures"]
        names = [str(s) for s in z["state_names"]] if "state_names" in z.files else \
            [f"state-{i + 1}" for i in range(z["M_site"].shape[0])]
        out["temperatures"] = [round(float(t), 6) for t in T]
        out["states"] = [{"name": nm, "M_abs_per_site": [round(float(v), 6) for v in z["M_site"][i]],
                          "energy_per_site": [round(float(v), 6) for v in z["E_site"][i]]}
                         for i, nm in enumerate(names)]
    return out


def _results(args):
    rec, d = _load(args.get("job"))
    b = rec["backend"]
    if b == "colab":
        s = summarize(d / "results")
        s["note"] = s.get("note", "") + ("; " if s.get("note") else "") + (
            "Colab runs are not fetched by this server: download the output folder from Colab (or through the "
            f"Colab MCP server) into {d / 'results'} and call results again")
        return {"job": _view(rec), "summary": s}
    if b != "local" and rec["state"] != "fetched":
        st = _status_of(rec, d)["job"]["state"]
        if st != "finished" and not (b == "runpod" and st.startswith("failed")):
            raise ToolError(f"job {rec['id']} is {st}: call results once it has finished")
        _fetch(rec, d, bool(args.get("keep")))
        _save(rec, d)
    elif b == "local":
        _status_of(rec, d)
    return {"job": _view(rec), "summary": summarize(Path(rec["results_dir"]))}


def _cancel(args):
    rec, d = _load(args.get("job"))
    b, note = rec["backend"], ""
    if b in ("local", "modal"):
        if b == "modal":
            m = re.search(r"ap-[A-Za-z0-9]{22}", _tail(d / "modal.log", 200000))
            if m:
                r = _sh(["modal", "app", "stop", "-y", m.group(0)], timeout=120)
                note = "modal app stopped" if r.returncode == 0 else f"modal app stop failed: {r.stderr[-500:]}"
            else:
                note = "no Modal app id in the log yet; stop it from modal.com/apps if it started"
        p = _PROCS.get(rec["id"])
        if (p is not None and p.poll() is None) or (p is None and _alive(rec.get("pid"))):
            try:
                os.killpg(int(rec["pid"]), signal.SIGTERM)
            except OSError:
                pass
        rec["state"] = "cancelled"
    elif b == "runpod":
        _runpod("DELETE", f"/pods/{rec['pod']}")
        rec.update(state="cancelled", pod_terminated=True)
    elif b == "kaggle":
        note = (f"the kaggle CLI cannot stop a running kernel: open {rec.get('url')} and press Cancel run "
                "(Kaggle also stops it at its time limit)")
    else:
        note = "Colab runs are not controlled by this server: stop the runtime in Colab"
    _save(rec, d)
    return {"job": _view(rec), "note": note}


# ── the pod side (RunPod) ────────────────────────────────────────────────────────────────────────

def make_server(root, token: str, port: int = 8000, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Serve only the files under root, only at /<token>/<path>; /<token>/ answers the file list as JSON.
    Anything else, the root and a wrong token included, is a bare 404 that lists nothing."""
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                  # the request line holds the token: never log it
            pass

        def _no(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            parts = urllib.request.unquote(self.path.split("?", 1)[0]).lstrip("/").split("/", 1)
            if len(parts) < 2 or not hmac.compare_digest(parts[0].encode(), token.encode()):
                return self._no()
            if parts[1] == "":
                files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) \
                    if root.exists() else []
                body = json.dumps({"files": files}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            target = (root / parts[1]).resolve()
            if root not in target.parents or not target.is_file():
                return self._no()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with open(target, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        do_HEAD = _no

    return ThreadingHTTPServer((host, port), Handler)


def _pod_main() -> int:
    """On the pod: write the run from SCK_CONFIG, serve its output folder behind SCK_TOKEN, run it with a time
    limit, mark DONE with the exit code and keep serving until the pod is terminated. It never exits on its own:
    RunPod would restart the container and run the whole simulation again."""
    cfg = json.loads(base64.b64decode(os.environ["SCK_CONFIG"]))
    token = os.environ.pop("SCK_TOKEN")
    out = Path(cfg.get("output_dir") or "simulation_output").resolve()
    out.mkdir(parents=True, exist_ok=True)
    srv = make_server(out, token, 8000)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    runfile.write(runfile.config(cfg), "run.py", "py")
    with open(out / "run.log", "wb") as log:
        try:
            rc = subprocess.run([sys.executable, "-u", "run.py"], stdout=log, stderr=subprocess.STDOUT,
                                timeout=float(os.environ.get("SCK_MAX_HOURS", "2")) * 3600,
                                env=dict(os.environ, MPLBACKEND="Agg")).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    (out / "DONE").write_text(str(rc))
    while True:                                       # results or cancel terminates the pod
        time.sleep(3600)


# ── registration ─────────────────────────────────────────────────────────────────────────────────

JOB = {"type": "string", "description": "the job id that run returned"}
mcp.register("backends", _backends, "Where a run can go: local, kaggle, modal, runpod, colab; for each, whether it "
             "is available and what to install or configure. Detects by presence only, never reads a credential.",
             {"type": "object", "properties": {}}, True)
mcp.register("run", _run, "Launch a run (checked first). Without confirm: true it only returns the plan: GPU, cost "
             "drivers, the time limit that bounds spending and the exact call to submit. Launch only what the user "
             "asked for. colab writes the notebook and hands off (no submission).",
             {"type": "object", "required": ["backend"], "properties": {
                 "config": mcp.CONFIG, "backend": {"type": "string", "enum": list(BACKENDS)},
                 "confirm": {"type": "boolean", "default": False},
                 "options": {"type": "object", "description": "local: max_hours. kaggle: accelerator, max_hours. "
                             "modal: gpu (default L4), max_hours (default 2, the function timeout). runpod: gpu "
                             "(default 'NVIDIA L4'), image, ref (package git ref, default v1.5.1), max_hours. A runpod pod "
                             "bills until results or cancel terminates it."}}}, False)
mcp.register("status", _status, "The state of a job: running, finished, failed; the log tail; for runpod the pod "
             "status and its cost per hour.", {"type": "object", "required": ["job"], "properties": {"job": JOB}},
             False)
mcp.register("results", _results, "Download a finished job's results if needed (a runpod pod is terminated after, "
             "unless keep: true) and summarize them: files and sizes, temperatures, per state |M| and energy per "
             "site against T, the summary.png path.",
             {"type": "object", "required": ["job"], "properties": {"job": JOB, "keep": {"type": "boolean",
                                                                                            "default": False}}}, False)
mcp.register("cancel", _cancel, "Stop a job: local kill, modal app stop, runpod pod terminate (kaggle: says how).",
             {"type": "object", "required": ["job"], "properties": {"job": JOB}}, False)


if __name__ == "__main__":
    if sys.argv[1:] == ["pod"]:
        sys.exit(_pod_main())
    sys.exit("usage: python -m Spin_Colour_Kernel.jobs pod   (the program a RunPod pod runs)")
