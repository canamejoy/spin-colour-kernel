"""Start the SCK MCP server from a clone of this repository, without installing the package.

    python3 sck_mcp.py

It needs only numpy to describe, check and write runs; running one on this machine also needs torch. The same
server, once the package is installed, is `python -m Spin_Colour_Kernel.mcp`. When numpy is missing it still
starts, with a single tool, `setup`, that tells the agent the exact command that installs it for this interpreter.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def _setup_only(missing: str) -> None:
    """A minimal MCP server for an interpreter without numpy: every call explains how to install it."""
    fix = f'"{sys.executable}" -m pip install numpy'
    text = (f"The SCK server needs numpy, which this Python ({sys.executable}) does not have ({missing}). "
            f"Ask the user before installing, then run: {fix}  and restart the sck MCP server.")
    tool = {"name": "setup", "description": "SCK cannot start yet: " + text,
            "inputSchema": {"type": "object", "properties": {}}}
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict) or msg.get("id") is None:
            continue
        method = msg.get("method")
        if method == "initialize":
            result = {"protocolVersion": (msg.get("params") or {}).get("protocolVersion") or "2025-06-18",
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "sck", "version": "setup"},
                      "instructions": text}
        elif method == "tools/list":
            result = {"tools": [tool]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": text}], "isError": True}
        elif method == "ping":
            result = {}
        else:
            out = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "method not found"}}
            print(json.dumps(out), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)


if __name__ == "__main__":
    try:
        import numpy  # noqa: F401
    except ImportError as e:
        _setup_only(str(e))
    else:
        from Spin_Colour_Kernel import mcp
        mcp.serve()
