"""The MCP server over stdio: the handshake, the tool list, and describe / check / write as an agent calls them."""

import json
import os
import subprocess
import sys
from pathlib import Path

from Spin_Colour_Kernel import mcp


def _session(*messages, cwd=None):
    lines = "\n".join(json.dumps(m) for m in messages) + "\n"
    src = str(Path(mcp.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-m", "Spin_Colour_Kernel.mcp"], input=lines, capture_output=True, text=True,
                       timeout=120, cwd=cwd, env=env)
    return [json.loads(l) for l in r.stdout.splitlines()]


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
READY = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _call(i, name, **arguments):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


def test_handshake_lists_the_tools_and_answers_only_requests():
    out = _session(INIT, READY, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert [m["id"] for m in out] == [1, 2]                     # the notification got no answer
    assert out[0]["result"]["protocolVersion"] == "2025-06-18"
    assert out[0]["result"]["serverInfo"]["name"] == "sck" and "tools" in out[0]["result"]["capabilities"]
    names = {t["name"] for t in out[1]["result"]["tools"]}
    assert {"describe", "check", "write"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in out[1]["result"]["tools"])


def test_check_and_write_through_the_protocol(tmp_path):
    cfg = {"crystal": "square", "box": [8, 8], "periodic": [True, True], "model": "ising",
           "states": [{"name": "a", "J": {"1": 1.0}}]}
    out = _session(INIT, _call(2, "check", config=cfg), _call(3, "write", config=cfg, format="py", path="run"),
                   _call(4, "write", config=cfg, format="py", path="run"), cwd=tmp_path)
    check = out[1]["result"]["structuredContent"]
    assert check["ok"] and check["sites"]["active"] == 64
    wrote = out[2]["result"]["structuredContent"]
    assert wrote["path"] == str(tmp_path / "run.py") and (tmp_path / "run.py").read_text().count('MODEL = "ising"') == 1
    assert out[3]["result"]["isError"] and "overwrite" in out[3]["result"]["content"][0]["text"]


def test_a_bad_configuration_is_a_tool_error_not_a_protocol_error(tmp_path):
    out = _session(INIT, _call(2, "write", config={"model": "potts"}), cwd=tmp_path)
    assert "error" not in out[1] and out[1]["result"]["isError"]
    assert "unknown model 'potts'" in out[1]["result"]["content"][0]["text"] and not list(tmp_path.iterdir())


def test_unknown_methods_and_tools_get_json_rpc_errors():
    assert mcp.handle({"jsonrpc": "2.0", "id": 5, "method": "nope"})["error"]["code"] == -32601
    r = mcp.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope"}})
    assert r["error"]["code"] == -32602
