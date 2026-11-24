"""MCP server: lets an agent prepare, check, write, launch and read SCK runs.

    python -m Spin_Colour_Kernel.mcp

It speaks the Model Context Protocol over stdio (JSON-RPC 2.0, one message per line) with the standard
library only. Register it in an agent, for example Claude Code:

    claude mcp add sck -- python -m Spin_Colour_Kernel.mcp

Tools: describe, check, write (this module), and the run tools of Spin_Colour_Kernel.jobs.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import agent, runfile

SERVER = {"name": "sck", "title": "SCK (Spin-Colour Kernel)", "version": "1.4.7"}
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "SCK runs classical spin Monte Carlo (Ising, XY, Heisenberg) on GPUs. Call describe for the settings, "
    "check to validate a configuration before anything else, then write to save a ready-to-run notebook or "
    "script, or the run tools to launch it where the user has access. Only launch a remote run the user asked for.")

CONFIG = {"type": "object", "description": "Run settings; any key left out takes its default (see describe).",
          "additionalProperties": True}


class ToolError(Exception):
    """A failure the agent should read and act on (reported with isError, not as a protocol error)."""


def _describe(args):
    return agent.describe()


def _check(args):
    return agent.check(args.get("config") or {})


def _write(args):
    r = agent.check(args.get("config") or {})
    if not r["ok"]:
        raise ToolError("the configuration has errors, nothing was written: " + "; ".join(r["errors"]))
    fmt = args.get("format") or "ipynb"
    if fmt not in runfile.FORMATS:
        raise ToolError(f"format must be one of {', '.join(runfile.FORMATS)}")
    path = Path(args.get("path") or f"spin_simulation.{fmt}").expanduser()
    if path.suffix != f".{fmt}":
        path = path.with_suffix(f".{fmt}")
    if path.exists() and not args.get("overwrite"):
        raise ToolError(f"{path} exists; pass overwrite: true to replace it")
    out = runfile.write(r["config"], path, fmt)
    return dict(path=str(out.resolve()), format=fmt, bytes=out.stat().st_size, warnings=r["warnings"],
                how_to_run="upload to Kaggle or Colab, or open in Jupyter, and press Run all" if fmt == "ipynb"
                else f"python {out.name} on a machine with an NVIDIA GPU (or the CPU, slowly)")


TOOLS = {
    "describe": (_describe, "The Hamiltonian, models, crystals, coordinate systems, state parameters, units, "
                 "file formats and the default value of every setting.",
                 {"type": "object", "properties": {}}, True),
    "check": (_check, "Validate a run configuration without running it: errors, warnings, active and surface "
              "sites, the colouring of every state, the temperature ladder and the size of the run.",
              {"type": "object", "properties": {"config": CONFIG}}, True),
    "write": (_write, "Write the run as a ready-to-run Jupyter notebook (.ipynb) or Python script (.py), the same "
              "file the SCK configurator page downloads. Refuses a configuration that check rejects.",
              {"type": "object", "properties": {
                  "config": CONFIG,
                  "format": {"type": "string", "enum": list(runfile.FORMATS), "default": "ipynb"},
                  "path": {"type": "string", "description": "file to write; default spin_simulation.<format>"},
                  "overwrite": {"type": "boolean", "default": False}}}, False),
}


def register(name, fn, description, schema, read_only=False):
    TOOLS[name] = (fn, description, schema, read_only)


def _tool_list():
    return [{"name": k, "description": d, "inputSchema": s,
             "annotations": {"readOnlyHint": ro, "openWorldHint": k in ("run", "status", "results", "cancel")}}
            for k, (_, d, s, ro) in TOOLS.items()]


def _call(params):
    name, args = params.get("name"), params.get("arguments") or {}
    if name not in TOOLS:
        raise KeyError(name)
    try:
        result = TOOLS[name][0](args)
    except ToolError as e:
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}
    except (ValueError, KeyError, TypeError, OSError) as e:
        return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    text = json.dumps(result, indent=1, default=str)
    return {"content": [{"type": "text", "text": text}], "structuredContent": json.loads(text)}


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message in, its response out (None for a notification)."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:                                   # notifications need no answer
        return None
    try:
        if method == "initialize":
            asked = (msg.get("params") or {}).get("protocolVersion")
            result = {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": SERVER, "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": _tool_list()}
        elif method == "tools/call":
            try:
                result = _call(msg.get("params") or {})
            except KeyError as e:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {e}"}}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    except Exception as e:                            # a bug here must not kill the session
        traceback.print_exc(file=sys.stderr)
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(stdin=None, stdout=None) -> None:
    """Answer messages from stdin on stdout. Anything else printed while a tool runs goes to stderr."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    sys.stdout = sys.stderr                           # a stray print must never corrupt the protocol stream
    try:
        from . import jobs  # noqa: F401  registers the run tools
    except ImportError:
        pass
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {e}"}}
        else:
            out = handle(msg) if isinstance(msg, dict) else {
                "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "batches are not supported"}}
        if out is not None:
            stdout.write(json.dumps(out) + "\n")
            stdout.flush()


if __name__ == "__main__":
    # Serve from the package module, not this __main__ copy, so the tools that jobs registers (into
    # Spin_Colour_Kernel.mcp) are the ones listed and called.
    from Spin_Colour_Kernel import mcp as _package_mcp
    _package_mcp.serve()
