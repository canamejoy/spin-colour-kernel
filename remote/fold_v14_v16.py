"""Fold the experiments v14-v16 runs into the product: tables and significance-v14-v16.json.

- E6metro (v14): generated kernel against CUDA-METRO, 2D triangular Heisenberg, H100 and A100-80GB.
- E6boris / E6boris-ext (v14): against BORIS, sc Heisenberg, 32^3-192^3.
- E5gen (v15): against the XLA baseline on held-out grids.
- E5genB (v16): against XLA at fixed geometry (sc 27^3), B = 1..64.

    python remote/fold_v14_v16.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "Spin-Colour-Kernel" / "Results"
RUNS = {
    "E6metro": {"H100": "h100/20261110-204853/E6metro.json", "A100-80GB": "a100/20261110-204852/E6metro.json"},
    "E6boris": {"H100": ["h100/20261110-210403/E6boris.json", "h100/20261110-213718/E6boris-ext.json"],
                "A100-80GB": ["a100/20261110-213210/E6boris.json", "a100/20261110-213721/E6boris-ext.json"]},
    "E5gen": {"H100": "h100/20261110-202634/E5gen.json", "A100-80GB": "a100/20261110-203015/E5gen.json"},
    "E5genB": {"H100": "h100/20261110-214710/E5genB.json", "A100-80GB": "a100/20261110-222433/E5genB.json"},
}


def load(p):
    return json.loads((R / p).read_text())


def write(name, header, rows):
    with (OUT / "tables" / name).open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def entry(comparison, lr, n, note, hyp, exp, name="log speed ratio (generated / competitor)"):
    return {"comparison": comparison, "test": "paired bootstrap of the log speed ratio, 95% percentile interval",
            "statistic": lr["mean"], "p_value": None, "p_adjusted": None, "correction": "none (interval reported)",
            "effect_size": lr["mean"], "effect_size_name": name, "n": n, "alpha": 0.05, "hypothesis": hyp,
            "experiment": exp, "interval": [lr["low"], lr["high"]], "note": note}


def r3(x):
    return round(float(x), 3)


def main():
    sig, rows = [], []
    for dev, p in RUNS["E6metro"].items():
        d = load(p)
        for r in d["rows"]:
            pc = r["p_choice"][0] if isinstance(r["p_choice"], list) else r["p_choice"]
            for b, prec in (("1", "float32"), ("1d", "float64")):
                lr = r[f"log_ratio_B{b}"]
                rows.append([dev, r["lattice"], r["L"], r["blocks_P"], pc, prec, r3(np.mean(r["ours_flips_per_ns"][b])),
                             round(float(np.mean(r["metro_flips_per_ns"])), 4), r3(lr["mean"]), r3(lr["low"]),
                             r3(lr["high"]), r3(math.exp(lr["mean"])), p.rsplit("/", 1)[0]])
                sig.append(entry(f"E6metro {dev} triangular L={r['L']} P={r['blocks_P']} B={b}", lr, d["reps"],
                                 f"CUDA-METRO {d['cudametro_commit'][:8]}, P choice {pc}; ours exact, CUDA-METRO "
                                 "approximate (0.0276 P/N); site index clamped (clamped_sites in the run)", "H5", "E6"))
    write("e6-cudametro.csv", ["Device", "Lattice", "L", "CUDA-METRO P", "P choice", "Ours precision", "Ours flips/ns",
                               "CUDA-METRO attempts/ns", "Log ratio", "Low", "High", "Speed-up", "Run"], rows)
    rows = []
    for dev, ps in RUNS["E6boris"].items():
        for p in ps:
            d = load(p)
            for r in d["rows"]:
                for b in ("1", "1d"):
                    prec = d["paired_boris_precision"][b] if isinstance(d.get("paired_boris_precision"), dict) else ("single" if b == "1" else "double")
                    lr = r[f"log_ratio_B{b}"]
                    rows.append([dev, "sc", r["cells"], r["active_sites"], prec, r3(np.mean(r["ours_flips_per_ns"][b])),
                                 r3(np.mean(r["boris_flips_per_ns"][prec])), r3(lr["mean"]), r3(lr["low"]), r3(lr["high"]),
                                 r3(math.exp(lr["mean"])), p.rsplit("/", 1)[0]])
                    sig.append(entry(f"E6boris {dev} sc {r['cells']}^3 B={b}", lr, d["reps"],
                                     f"BORIS {d['boris_commit'][:8]} ({prec} build, Linux size_t atomic patch); "
                                     "cone move adapting to 50% acceptance against ours uniform on the sphere",
                                     "H5", "E6"))
    write("e6-boris.csv", ["Device", "Crystal", "Cells", "Sites", "Precision (both)", "Ours flips/ns", "BORIS flips/ns",
                           "Log ratio", "Low", "High", "Speed-up", "Run"], rows)
    rows = []
    for dev, p in RUNS["E5gen"].items():
        d = load(p)
        v = d["summary"]["v15"]
        for g in v["grids"]:
            rows.append([dev, g["factor"], g["B"], g["n"], g["b"], "x".join(map(str, g["box"])), g["generated_periodic"],
                         r3(g["log_speedup"]), r3(g["low"]), r3(g["high"]), r3(math.exp(g["log_speedup"])),
                         g["generated_faster"], p.rsplit("/", 1)[0]])
            sig.append(entry(f"E5gen {dev} {g['factor']} B={g['B']} n={g['n']} b={g['b']} box={'x'.join(map(str, g['box']))}",
                             {"mean": g["log_speedup"], "low": g["low"], "high": g["high"]}, 10,
                             "generated against the XLA baseline, interleaved rounds", "H4", "E5",
                             "log speed-up (XLA time / generated time)"))
        for fac, s in v["spread"].items():
            sig.append({"comparison": f"E5gen {dev} {fac} cost spread over L1", "test": "max / min of the median cost",
                        "statistic": s["xla_max_over_min"], "p_value": None, "p_adjusted": None,
                        "correction": "none (descriptive)", "effect_size": s["generated_max_over_min"],
                        "effect_size_name": "generated max / min (statistic: XLA max / min)", "n": len(s["L1"]),
                        "alpha": None, "hypothesis": "H4", "experiment": "E5", "note": "L1 = 99..109"})
    write("e5-gen-vs-xla.csv", ["Device", "Grid set", "B", "n", "b", "Box", "Generated periodic", "Log speed-up",
                                "Low", "High", "Speed-up", "Generated faster", "Run"], rows)
    rows = []
    for dev, p in RUNS["E5genB"].items():
        d = load(p)
        v = d["summary"]["v16"]
        sp = {(g["factor"], g["B"]): g for g in v["speedup"]["grids"]}
        for fac, m in v["per_model"].items():
            for i, B in enumerate(m["B"]):
                g = sp[(fac, B)]
                rows.append([dev, "Heisenberg" if fac.endswith("heis") else "Ising", B, r3(m["xla_median_ns"][i]),
                             r3(m["generated_median_ns"][i]), r3(math.exp(g["log_speedup"])), r3(g["low"]), r3(g["high"]),
                             p.rsplit("/", 1)[0]])
                sig.append(entry(f"E5genB {dev} {fac} B={B}", {"mean": g["log_speedup"], "low": g["low"], "high": g["high"]},
                                 10, "sc 27^3, generated against XLA, interleaved rounds", "H4", "E5",
                                 "log speed-up (XLA time / generated time)"))
            sig.append({"comparison": f"E5genB {dev} {fac} upward jumps", "test": "count of B -> next B rises above 5%",
                        "statistic": m["xla_upward_jumps"], "p_value": None, "p_adjusted": None,
                        "correction": "none (descriptive)", "effect_size": m["generated_upward_jumps"],
                        "effect_size_name": "generated upward jumps (statistic: XLA upward jumps)", "n": len(m["B"]),
                        "alpha": None, "hypothesis": "H4", "experiment": "E5",
                        "note": f"max jump XLA {math.exp(m['xla_max_jump']):.3f}x, generated {math.exp(m['generated_max_jump']):.3f}x"})
    write("e5-genb-replicas.csv", ["Device", "Model", "B", "XLA ns/update", "Generated ns/update", "Speed-up",
                                   "Log low", "Log high", "Run"], rows)
    for i in range(0, len(sig), 200):
        (OUT / f"significance-v14-v16-{i // 200 + 1}.json").write_text(json.dumps({"tests": sig[i:i + 200]}, indent=2))
    print(len(sig), "tests")


if __name__ == "__main__":
    main()
