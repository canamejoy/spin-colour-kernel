"""The run tools: plans that submit nothing, credential hygiene, the RunPod request and token server, local jobs.

Nothing here contacts Kaggle, Modal, RunPod or Colab, submits a remote run or starts a GPU run: subprocess and
urllib are replaced by fakes that record what would have been sent."""

import base64
import json
import subprocess
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from Spin_Colour_Kernel import jobs, mcp

SECRET = "sentinel-secret-9f8e7d6c5b4a"
CFG = {"crystal": "square", "box": [8, 8], "periodic": [True, True], "model": "ising",
       "states": [{"name": "a", "J": {"1": 1.0}}], "T_max": 2.0, "T_min": 1.0, "T_step": 0.5}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An isolated HOME and jobs folder; every credential holds the sentinel; no network, no subprocess."""
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text(json.dumps({"username": "alice", "key": SECRET}))
    (home / ".modal.toml").write_text(f'[default]\ntoken_id = "{SECRET}"\ntoken_secret = "{SECRET}"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SCK_JOBS_DIR", str(tmp_path / "jobs"))
    for k in ("KAGGLE_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "RUNPOD_API_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.setenv(k, SECRET)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MODAL_CONFIG_PATH", raising=False)
    calls = []

    def forbidden(*a, **k):
        calls.append(a)
        raise AssertionError(f"unexpected external call {a[:1]}")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(jobs.shutil, "which", lambda name: f"/usr/bin/{name}")
    return tmp_path, calls


def _call(name, **arguments):
    r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}})
    return r["result"]


def _everything_written(root):
    return "".join(p.read_text(errors="replace") for p in root.rglob("*") if p.is_file() and p.suffix != ".npz")


def test_tools_list_includes_the_run_tools():
    r = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = {t["name"]: t for t in r["result"]["tools"]}
    assert {"backends", "run", "status", "results", "cancel", "describe", "check", "write"} <= set(tools)
    assert tools["backends"]["annotations"]["readOnlyHint"] and tools["run"]["annotations"]["openWorldHint"]


@pytest.mark.parametrize("backend", ["kaggle", "modal", "runpod", "local"])
def test_a_plan_without_confirm_submits_nothing(env, backend):
    tmp, calls = env
    r = _call("run", config=CFG, backend=backend)
    assert not r.get("isError"), r
    out = r["structuredContent"]
    assert out["submitted"] is False and calls == [] and not (tmp / "jobs").exists()
    plan = out["plan"]
    assert plan["cost_drivers"]["chains"] == 32 and plan["cost_drivers"]["temperatures"] == 3
    assert plan["next_call"]["arguments"]["confirm"] is True and "time_limit" in plan and plan["gpu"]
    assert SECRET not in json.dumps(r)


def test_a_bad_configuration_is_refused_before_anything_else(env):
    r = _call("run", config={"model": "potts"}, backend="runpod", confirm=True)
    assert r["isError"] and "nothing was launched" in r["content"][0]["text"] and env[1] == []


def test_detection_never_returns_a_credential_value(env):
    r = _call("backends")
    text = json.dumps(r)
    assert SECRET not in text
    b = r["structuredContent"]
    assert b["kaggle"]["available"] and b["kaggle"]["user"] == "alice"
    assert b["modal"]["available"] and b["runpod"]["available"] and b["colab"]["available"]
    assert env[1] == []


def test_backends_works_with_nothing_installed_or_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR", "MODAL_TOKEN_ID",
              "MODAL_TOKEN_SECRET", "MODAL_CONFIG_PATH", "RUNPOD_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(jobs.shutil, "which", lambda name: None)
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: None)
    b = jobs.detect()
    assert not any(b[k]["available"] for k in ("local", "kaggle", "modal", "runpod")) and b["colab"]["available"]
    assert "pip install kaggle" in b["kaggle"]["what_to_install_or_configure"]
    assert "RUNPOD_API_KEY" in b["runpod"]["what_to_install_or_configure"]


def test_runpod_request_uses_v2_fields_and_never_carries_the_key(env, monkeypatch):
    tmp, _ = env
    sent = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        sent.append(req)
        return Resp({"id": "pod123", "status": "PROVISIONING"})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    r = _call("run", config=CFG, backend="runpod", confirm=True, options={"gpu": "NVIDIA L4"})
    assert not r.get("isError"), r
    (req,) = sent
    assert req.full_url == "https://api.runpod.io/v2/pods" and req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {SECRET}"           # the key travels only here
    body = json.loads(req.data)
    assert {"name", "image", "gpu", "disk", "ports", "env", "cmd"} <= set(body)
    assert body["gpu"] == {"id": "NVIDIA L4", "count": 1} and body["ports"] == ["8000/http"]
    assert SECRET not in req.data.decode()
    assert json.loads(base64.b64decode(body["env"]["SCK_CONFIG"]))["model"] == "ising"
    assert "git+https://github.com/canamejoy/spin-colour-kernel@v1.5.0" in body["cmd"][-1]
    job = r["structuredContent"]["job"]
    assert job["pod"] == "pod123" and "token" not in job
    assert SECRET not in _everything_written(tmp / "jobs") and SECRET not in json.dumps(r)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_token_server_serves_only_under_the_token(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "DONE").write_text("0")
    (tmp_path / "secret.txt").write_text("outside")
    srv = jobs.make_server(tmp_path / "out", "tok-abc", 0, "127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for path in ("/", "/DONE", "/wrong/", "/wrong/DONE", "/tok-ab/DONE", "/tok-abc", "/tok-abc/../secret.txt",
                     "/tok-abc/%2e%2e/secret.txt"):
            code, body = _get(base + path)
            assert code == 404 and body == b"", path
        assert _get(base + "/tok-abc/DONE") == (200, b"0")
        code, body = _get(base + "/tok-abc/")
        assert code == 200 and json.loads(body) == {"files": ["DONE"]}
    finally:
        srv.shutdown()


def _fake_local_job(tmp, state_exit="0"):
    job = "local-20261123-101500-a1b2c3"
    d = tmp / "jobs" / job
    out = d / "simulation_output"
    out.mkdir(parents=True)
    rec = dict(id=job, backend="local", pid=2 ** 22 + 7, state="running", output_dir="simulation_output",
               results_dir=str(out), options={"max_hours": 1})
    (d / "job.json").write_text(json.dumps(rec))
    (d / "run.log").write_text("total: 0.10 min\n")
    if state_exit is not None:
        (d / "EXIT").write_text(state_exit)
    return job, d, out


def test_local_status_and_results_on_a_finished_job(env):
    tmp, calls = env
    job, d, out = _fake_local_job(tmp)
    np.savez(out / "results.npz", temperatures=np.array([2.0, 1.5]), state_names=np.array(["a", "b"]),
             M_site=np.array([[0.1, 0.5], [0.2, 0.6]]), E_site=np.array([[-0.5, -1.0], [-0.6, -1.1]]))
    (out / "summary.png").write_bytes(b"png")
    (out / "configurations.npy").write_bytes(b"not an array: must never be loaded")
    st = _call("status", job=job)["structuredContent"]
    assert st["job"]["state"] == "finished" and "total" in st["log_tail"]
    s = _call("results", job=job)["structuredContent"]["summary"]
    assert s["temperatures"] == [2.0, 1.5] and s["summary_png"].endswith("summary.png")
    assert s["states"][1] == {"name": "b", "M_abs_per_site": [0.2, 0.6], "energy_per_site": [-0.6, -1.1]}
    assert {f["path"] for f in s["files"]} >= {"results.npz", "configurations.npy"} and calls == []


def test_local_status_failed_and_stopped(env):
    tmp, _ = env
    job, d, _ = _fake_local_job(tmp, "3")
    assert _call("status", job=job)["structuredContent"]["job"]["state"] == "failed (exit 3)"
    (d / "EXIT").unlink()
    assert _call("status", job=job)["structuredContent"]["job"]["state"].startswith("failed (stopped")


def test_local_run_starts_detached_in_the_job_folder(env, monkeypatch):
    tmp, _ = env
    started = []

    class P:
        pid, returncode = 4242, None

        def __init__(self, cmd, **kw):
            started.append((cmd, kw))

        def poll(self):
            return None
    monkeypatch.setattr(subprocess, "Popen", P)
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: object())
    r = _call("run", config=CFG, backend="local", confirm=True)["structuredContent"]
    (cmd, kw), = started
    d = tmp / "jobs" / r["job"]["id"]
    assert kw["start_new_session"] and kw["cwd"] == d and (d / "run.py").exists()
    assert r["job"]["pid"] == 4242 and _call("status", job=r["job"]["id"])["structuredContent"]["job"]["state"] == "running"


def test_colab_writes_the_notebook_and_hands_off(env):
    tmp, calls = env
    r = _call("run", config=CFG, backend="colab")["structuredContent"]
    assert r["submitted"] is False and calls == []
    assert (tmp / "jobs" / r["job"]["id"] / "run.ipynb").exists() and "colab-mcp" in r["next"]
    st = _call("status", job=r["job"]["id"])["structuredContent"]
    assert "not tracked" in st["note"]
    assert "not fetched" in _call("results", job=r["job"]["id"])["structuredContent"]["summary"]["note"]


def test_kaggle_push_builds_a_private_gpu_notebook_kernel(env, monkeypatch):
    tmp, _ = env
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        out = "usage: --accelerator ACC --timeout" if "--help" in cmd else "Kernel version 1 successfully pushed"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = _call("run", config=CFG, backend="kaggle", confirm=True)["structuredContent"]
    d = tmp / "jobs" / r["job"]["id"]
    meta = json.loads((d / "kernel" / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["kernel_type"] == "notebook" and meta["id"].startswith("alice/sck-kaggle-")
    assert ran[-1][:5] == ["kaggle", "kernels", "push", "-p", str(d / "kernel")] and "--accelerator" in ran[-1]
    assert SECRET not in _everything_written(tmp / "jobs")


def test_modal_submits_detached_with_a_bounded_timeout(env, monkeypatch):
    tmp, _ = env
    started = []

    class P:
        pid, returncode = 77, None

        def __init__(self, cmd, **kw):
            started.append((cmd, kw))
    monkeypatch.setattr(subprocess, "Popen", P)
    r = _call("run", config=CFG, backend="modal", confirm=True, options={"gpu": "A10G", "max_hours": 1.5})
    job = r["structuredContent"]["job"]
    (cmd, kw), = started
    app = (tmp / "jobs" / job["id"] / "modal_app.py").read_text()
    assert cmd == ["modal", "run", "-d", "modal_app.py"] and kw["start_new_session"]
    assert "'A10G'" in app and "5400" in app and "sck-runs" in app and SECRET not in app


def test_unknown_jobs_and_options_are_tool_errors(env):
    assert _call("status", job="../etc")["isError"]
    assert _call("run", config=CFG, backend="modal", options={"gpus": 2})["isError"]


@pytest.mark.parametrize("confirm", ["false", "true", 1, "yes"])
def test_only_a_real_true_confirms_a_launch(env, confirm):
    tmp, calls = env
    out = _call("run", config=CFG, backend="runpod", confirm=confirm)["structuredContent"]
    assert out["submitted"] is False and calls == [] and not (tmp / "jobs").exists()


def test_the_runpod_plan_says_the_pod_bills_until_it_is_terminated(env):
    plan = _call("run", config=CFG, backend="runpod")["structuredContent"]["plan"]
    assert "until results or cancel terminates it" in plan["billing"]
    assert "keeps billing" in plan["time_limit"]
    r = _call("run", config=CFG, backend="runpod", options={"serve_hours": 1})
    assert r["isError"] and "unknown option 'serve_hours'" in r["content"][0]["text"]


def test_the_pod_program_never_exits_after_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = dict(CFG, output_dir="out")
    monkeypatch.setenv("SCK_CONFIG", base64.b64encode(json.dumps(cfg).encode()).decode())
    monkeypatch.setenv("SCK_TOKEN", "t" * 32)

    class Server:
        def serve_forever(self):
            pass
    monkeypatch.setattr(jobs, "make_server", lambda *a, **k: Server())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))

    class StillServing(Exception):
        pass
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            raise StillServing
    monkeypatch.setattr(jobs.time, "sleep", sleep)
    with pytest.raises(StillServing):                 # it would have kept sleeping: RunPod never restarts it
        jobs._pod_main()
    assert (tmp_path / "out" / "DONE").read_text() == "0"


def _fake_pod_job(tmp):
    job = "runpod-20261123-101500-a1b2c3"
    d = tmp / "jobs" / job
    d.mkdir(parents=True)
    rec = dict(id=job, backend="runpod", pod="pod123", proxy="https://pod123-8000.proxy.runpod.net",
               token="t" * 32, state="finished", output_dir="simulation_output", options={"max_hours": 1})
    (d / "job.json").write_text(json.dumps(rec))
    return job, d


def test_a_failed_download_says_the_pod_is_still_billing_and_does_not_terminate_it(env, monkeypatch):
    tmp, _ = env
    job, _ = _fake_pod_job(tmp)
    sent = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        sent.append("GET" if isinstance(req, str) else req.get_method())
        if "api.runpod.io" in url:
            return Resp(json.dumps({"id": "pod123", "status": "RUNNING"}).encode())
        if url.endswith("/DONE"):
            return Resp(b"0")
        raise urllib.error.URLError("proxy 502")              # the file listing and downloads fail
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    r = _call("results", job=job)
    assert r["isError"] and "still running and billing" in r["content"][0]["text"]
    assert "DELETE" not in sent


def test_an_exited_pod_is_reported_as_exited(env, monkeypatch):
    tmp, _ = env
    job, _ = _fake_pod_job(tmp)

    class Resp:
        def read(self):
            return json.dumps({"id": "pod123", "status": "EXITED", "cost": 0.4}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Resp())
    st = _call("status", job=job)["structuredContent"]
    assert st["job"]["state"] == "pod exited" and st["pod_status"] == "EXITED"


def test_a_cancelled_local_job_stays_cancelled(env):
    tmp, _ = env
    job, d, _ = _fake_local_job(tmp, None)
    rec = json.loads((d / "job.json").read_text())
    (d / "job.json").write_text(json.dumps(dict(rec, state="cancelled")))
    assert _call("status", job=job)["structuredContent"]["job"]["state"] == "cancelled"


@pytest.mark.parametrize("stdout,state", [("0✓ Finished downloading files to local!\n", "finished"),
                                          ("3\n✓ Finished downloading files to local!", "failed (exit 3)")])
def test_modal_status_reads_the_exit_code_past_the_cli_progress_line(env, monkeypatch, stdout, state):
    tmp, _ = env
    job = "modal-20261123-222258-38636e"
    d = tmp / "jobs" / job
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps(dict(id=job, backend="modal", state="submitted", pid=2 ** 22 + 9,
                                                output_dir="simulation_output", options={"gpu": "L4"})))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout, ""))
    assert _call("status", job=job)["structuredContent"]["job"]["state"] == state
