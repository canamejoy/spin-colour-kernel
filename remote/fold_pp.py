"""Fold the periodic-against-periodic E5pp runs into results-product tables.

usage: fold_pp.py <out-dir> <device>=<run-dir> [<device>=<run-dir> ...]
Writes e5pp-gen-vs-xla.csv (one row per grid and device: paired log speed-up of the generated kernel over the
XLA baseline with its bootstrap interval, both sides periodic) and e5pp-spread.csv (max / min median cost over
the slab sweeps for each side).
"""
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
pairs = [a.split("=", 1) for a in sys.argv[2:]]
grid_rows, spread_rows = [], []
for device, run in pairs:
    d = json.loads((Path(run) / "E5pp.json").read_text())
    v = d["summary"]["v15"]
    tag = "/".join(Path(run).parts[-2:])
    for g in v["grids"]:
        grid_rows.append({"Device": device, "Grid set": g["factor"], "B": g["B"], "n": g["n"], "b": g["b"],
                          "Box": "x".join(map(str, g["box"])), "Generated periodic": g["generated_periodic"],
                          "Log speed-up": round(g["log_speedup"], 3), "Low": round(g["low"], 3),
                          "High": round(g["high"], 3), "Speed-up": round(math.exp(g["log_speedup"]), 3),
                          "Generated faster": g["generated_faster"], "Run": tag})
    for fac, s in v["spread"].items():
        spread_rows.append({"Device": device, "Grid set": fac, "L1": " ".join(map(str, s["L1"])),
                            "XLA max/min": round(s["xla_max_over_min"], 3),
                            "Generated max/min": round(s["generated_max_over_min"], 3), "Run": tag})
for name, rows in (("e5pp-gen-vs-xla.csv", grid_rows), ("e5pp-spread.csv", spread_rows)):
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(name, len(rows))
