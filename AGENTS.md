# Working with spin-colour-kernel as an agent

Most people who open this repository through an agent want to simulate a spin model, not to develop the package.
Help them through the SCK MCP server, which this repository declares in `.mcp.json`.

## Start the server

- Claude Code asks once whether to use the `sck` server from `.mcp.json`; approve it. The server runs
  `python3 sck_mcp.py` from the repository root (set `SCK_PYTHON` if the interpreter has another name, e.g. `python`
  on Windows).
- With `pip install spin-colour-kernel` the same server is the `sck-mcp` command, and Python code can call `import sck` (`sck.check`, `sck.write`, `sck.run`).
- Describing, checking and writing a run needs only `numpy` (`pip install numpy`). Running it on this machine also
  needs `torch` (`pip install torch matplotlib`).
- Another agent: register `python3 /path/to/the/clone/sck_mcp.py` as a stdio MCP server. Without MCP, the same
  functions are in `src/Spin_Colour_Kernel/agent.py` (`describe`, `check`) and `runfile.py` (`write`).

## Workflow

1. `describe` for the vocabulary: models, crystals, coordinate systems, state parameters, defaults.
2. Translate what the person asks into a configuration, then `check` it and fix every error before going on.
   Tell them what `check` reports (active and surface sites, colours, temperatures, run size).
3. `write` a `.ipynb` (Kaggle, Colab, Jupyter) or `.py` (their own machine) when they want a file.
4. To launch: `backends` shows what this machine can reach. `run` without `confirm` returns a plan; show it to
   the person, and call `run` again with `confirm: true` only after they agree. Remote runs spend their money or
   quota; a RunPod pod bills until `results` or `cancel` terminates it.
5. `status`, then `results` for the summary (temperatures, |M| and energy per site per state, `summary.png`).

Never read, print or ask for the contents of credential files or keys. The configurator page
`ready_for_run/configurator.html` does steps 1 to 3 in a browser, with the lattice drawn.
