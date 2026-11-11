"""Fold the experiments v17 runs (chunked long rows): E1chunk exactness on the GPU and the CUDA-METRO repeat.

    python remote/fold_v17.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "Spin-Colour-Kernel" / "Results"
RUNS = {"H100": "h100/20261111-000831", "A100-80GB": "a100/20261111-000831"}


def write(name, header, rows):
    with (OUT / "tables" / name).open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main():
    sig, ex, mt = [], [], []
    for dev, run in RUNS.items():
        c = json.loads((R / run / "E1chunk.json").read_text())
        for r in c["rows"]:
            ex.append([dev, r["system"], "x".join(map(str, r["box"])), r["sites"], r["chunks"], r["sweeps"],
                       r["differing_sites"], round(r["chunked_flips_per_ns"], 3), round(r["unchunked_flips_per_ns"], 3), run])
            sig.append({"comparison": f"E1chunk {dev} {r['system']} {'x'.join(map(str, r['box']))}",
                        "test": "spin-by-spin comparison, chunked against one-program-per-row kernel",
                        "statistic": r["differing_sites"], "p_value": None, "p_adjusted": None,
                        "correction": "none (deterministic)", "effect_size": r["differing_sites"],
                        "effect_size_name": "differing sites after the sweeps", "n": r["sites"], "alpha": None,
                        "hypothesis": "H1", "experiment": "E1", "note": f"{r['chunks']} chunks, {r['sweeps']} sweeps (v17)"})
        m = json.loads((R / run / "E6metro-v17.json").read_text())
        for r in m["rows"]:
            pc = r["p_choice"][0] if isinstance(r["p_choice"], list) else r["p_choice"]
            for b, prec in (("1", "float32"), ("1d", "float64")):
                lr = r[f"log_ratio_B{b}"]
                mt.append([dev, r["L"], r["blocks_P"], pc, prec, round(float(np.mean(r["ours_flips_per_ns"][b])), 3),
                           round(float(np.mean(r["metro_flips_per_ns"])), 4), round(lr["mean"], 3), round(lr["low"], 3),
                           round(lr["high"], 3), round(math.exp(lr["mean"]), 3), run])
                sig.append({"comparison": f"E6metro v17 {dev} triangular L={r['L']} P={r['blocks_P']} B={b}",
                            "test": "paired bootstrap of the log speed ratio, 95% percentile interval",
                            "statistic": lr["mean"], "p_value": None, "p_adjusted": None,
                            "correction": "none (interval reported)", "effect_size": lr["mean"],
                            "effect_size_name": "log speed ratio (generated / competitor)", "n": m["reps"],
                            "alpha": 0.05, "hypothesis": "H5", "experiment": "E6", "interval": [lr["low"], lr["high"]],
                            "note": f"v17 chunked kernel; P choice {pc}"})
    write("e1-chunk-exactness.csv", ["Device", "System", "Box", "Sites", "Chunks", "Sweeps", "Differing sites",
                                     "Chunked flips/ns", "Unchunked flips/ns", "Run"], ex)
    write("e6-cudametro-v17.csv", ["Device", "L", "CUDA-METRO P", "P choice", "Ours precision", "Ours flips/ns",
                                   "CUDA-METRO attempts/ns", "Log ratio", "Low", "High", "Speed-up", "Run"], mt)
    (OUT / "significance-v17.json").write_text(json.dumps({"tests": sig}, indent=2))
    print(len(sig), "tests")


if __name__ == "__main__":
    main()
