"""Fold the fetched Kaggle outputs (results/kaggle/<job>/results/*.json) into the product folder
Spin-Colour-Kernel/Results/ that the papersmith dashboard reads: tables (CSV), significance.json and
figures. Every number is copied from the JSON files; nothing is recomputed except plotting.

    python remote/build_product.py
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "results" / "kaggle"
OUT = ROOT / "Spin-Colour-Kernel" / "Results"
JOBS = {"E1": "a", "E3": "a", "E2": "e", "E2b": "f", "E6": "b", "E8": "d", "E5": "c",
        "E5-jax0435": "h", "E5-sass": "g"}


def load(name):
    return json.loads((K / JOBS[name] / "results" / f"{name}.json").read_text())


def write_csv(name, header, rows):
    p = OUT / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sig = []
    # E1
    e1 = load("E1")
    write_csv("e1-exactness.csv", ["System", "Colours", "Active sites", "Seeds run", "Sweeps compared",
                                   "Differing spins", "First failing sweep"],
              [[r["system"], r["colours"], r["active_sites"], r["seeds"], r["sweeps_compared"],
                r["differing_spins"], (r["first_failure"] or {}).get("sweep", "none")] for r in e1["rows"]])
    # E3
    e3 = load("E3")
    write_csv("e3-colourings.csv", ["Crystal and shells", "Group", "Linear colours", "Clique number", "Admissible",
                                    "Optimality certified"],
              [[f"{r['crystal']}, {'-'.join(map(str, r['shells']))}", "x".join(f"Z{m}" for m in r["group"]),
                r["linear_colours"], r["clique_number"], r["admissible"], r["optimality_certified"]] for r in e3["rows"]])
    # E2 and E2b
    rows = []
    for tag in ("E2", "E2b"):
        d = load(tag)
        acc = d["params"].get("acceptance", "metropolis")
        for r in d["rows"]:
            rows.append([tag, r["lattice"], r["rule"], acc, len(r["seeds"]), r["rejected_seeds"],
                         round(r["median_chi2"], 3), round(r.get("median_cohens_w", np.median([np.sqrt(x["chi2"] / d["params"]["samples"]) for x in r["seeds"]])), 5)])
            for s in r["seeds"]:
                sig.append({"comparison": f"{tag} {r['lattice']} {r['rule']} ({acc}) seed {s['seed']}",
                            "test": "chi-square goodness of fit against exact enumeration",
                            "statistic": s["chi2"], "p_value": s["p"], "p_adjusted": s["p_holm"],
                            "correction": "Holm over 10 seeds", "effect_size": float(np.sqrt(s["chi2"] / d["params"]["samples"])),
                            "effect_size_name": "Cohen's w (goodness of fit, sqrt(chi2/n))", "n": d["params"]["samples"], "alpha": 0.01,
                            "hypothesis": "H1b", "experiment": tag})
    write_csv("e2-colouring-control.csv", ["Run", "Lattice", "Rule", "Acceptance", "Seeds", "Seeds rejected (Holm)",
                                           "Median chi-square", "Median Cohen's w"], rows)
    # E6
    e6 = load("E6")
    rows = []
    for r in e6["slopes"]:
        B = [str(b) for b in r["batches"]]
        rows.append([r["system"], r["active_sites"]] + [round(float(np.mean(r["flips_per_ns"][b])), 3) for b in B]
                    + [round(r["log_cost_slope_per_doubling"], 4), round(r["slope_low"], 4), round(r["slope_high"], 4)])
        sig.append({"comparison": f"E6 batch slope {r['system']}", "test": "bootstrap of the log-cost slope per doubling",
                    "statistic": r["log_cost_slope_per_doubling"], "ci_low": r["slope_low"], "ci_high": r["slope_high"],
                    "n": 10, "alpha": 0.05, "hypothesis": "H5", "experiment": "E6"})
    write_csv("e6-throughput-t4.csv", ["System", "Active sites"] + [f"Flips/ns B={b}" for b in e6["slopes"][0]["batches"]]
              + ["Log-cost slope", "Slope low", "Slope high"], rows)
    p = e6["pair"]
    write_csv("e6-ising-gpu-pair.csv", ["Configuration and device", "Generated flips/ns", "ising-gpu spins/ns", "Log ratio",
                                        "Interval low", "Interval high"],
              [["2D Ising 4096^2 at Tc, T4", round(float(np.mean(p["generated_flips_per_ns"])), 3),
                round(float(np.mean(p["ising_gpu_spins_per_ns"])), 3), round(p["log_ratio"], 4), round(p["low"], 4),
                round(p["high"], 4)]])
    sig.append({"comparison": "E6 ising-gpu pair", "test": "paired bootstrap of the log speed ratio",
                "statistic": p["log_ratio"], "ci_low": p["low"], "ci_high": p["high"], "n": 10, "alpha": 0.05,
                "hypothesis": "H5", "experiment": "E6"})
    # E8
    e8 = load("E8")
    write_csv("e8-two-gpu.csv", ["System", "Total batch", "1 GPU flips/ns", "2 GPU flips/ns", "Scaling", "Low", "High"],
              [[r["system"], r["batch_total"], round(float(np.mean(r["flips_per_ns"]["1"])), 3),
                round(float(np.mean(r["flips_per_ns"]["2"])), 3)] + [round(x, 3) for x in r["scaling"]]
               for r in e8["rows"]])
    for r in e8["rows"]:
        sig.append({"comparison": f"E8 two-GPU scaling {r['system']}", "test": "bootstrap of the scaling ratio",
                    "statistic": r["scaling"][0], "ci_low": r["scaling"][1], "ci_high": r["scaling"][2], "n": 10,
                    "alpha": 0.05, "experiment": "E8"})
    # E5
    rows = []
    for tag, label in (("E5", "JAX 0.11.1, LLVM IR of the MLIR emitters"), ("E5-jax0435", "JAX 0.4.35, PTX dump"),
                       ("E5-sass", "JAX 0.11.1, SASS from the compilation cache")):
        d = load(tag)
        base = {"B": lambda r: r["n"] in (1, 3) and r["b"] == 1 and r["box"][0] == 5,
                "n": lambda r: r["B"] in (1, 3) and r["b"] == 1 and r["box"][0] == 5,
                "b": lambda r: r["B"] in (1, 3) and r["n"] in (1, 3) and r["box"][0] == 5,
                "box": lambda r: r["B"] in (1, 3) and r["n"] in (1, 3) and r["b"] == 1,
                "all": lambda r: True}
        names = {"B": "Batch size", "n": "Spin components", "b": "Basis size", "box": "Box extents", "all": "All grids"}
        for fac, sel in base.items():
            rs = [r for r in d["rows"] if sel(r)]
            un = sum(r["measured_module"] is None for r in rs)
            ag = sum(r["measured_module"] == r["predicted"] for r in rs)
            rows.append([label, names[fac], len(rs), ag, len(rs) - ag - un, un])
    write_csv("e5-width-law.csv", ["Compiler and reading", "Factor varied", "Grids", "Agreeing grids",
                                   "Disagreeing grids", "Unreadable"], rows)
    (OUT / "significance.json").write_text(json.dumps({"tests": sig}, indent=2))
    figures()
    print("product results written:", OUT, len(sig), "tests")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # E2 histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for j, tag in enumerate(("E2", "E2b")):
        d = load(tag)
        for i, lat in enumerate(("Ising square", "Ising triangular")):
            ax = axes[i, j]
            h = d["histograms"][f"{lat}|coloured"]
            hs = d["histograms"][f"{lat}|all sites"]
            x = np.arange(len(h["levels"]))
            ax.bar(x - 0.27, h["exact"], 0.27, label="exact", color="#888")
            ax.bar(x, h["sampled"], 0.27, label="coloured kernel", color="#2a7ab9")
            ax.bar(x + 0.27, hs["sampled"], 0.27, label="all sites at once", color="#d9822b")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.5)
            ax.set_xticks(x, [f"{v:g}" for v in h["levels"]], rotation=90, fontsize=7)
            ax.set_title(f"{lat} ({'Metropolis' if tag == 'E2' else 'Barker'} acceptance)", fontsize=10)
            ax.set_xlabel("energy")
            if j == 0:
                ax.set_ylabel("probability")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(fig_dir / "e2-histograms.png", dpi=130)
    plt.close(fig)
    # E6 throughput against batch
    e6 = load("E6")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for r in e6["slopes"]:
        B = r["batches"]
        ax.plot(B, [np.mean(r["flips_per_ns"][str(b)]) for b in B], marker="o", label=r["system"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("replica batch B")
    ax.set_ylabel("spin updates per ns (T4)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(fig_dir / "e6-throughput.png", dpi=130)
    plt.close(fig)
    # E5 measured against predicted width
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for tag, mk, lab in (("E5", "o", "JAX 0.11.1"), ("E5-jax0435", "x", "JAX 0.4.35")):
        d = load(tag)
        pr = np.array([r["predicted"] for r in d["rows"]])
        me = np.array([r["measured_module"] or 0 for r in d["rows"]])
        jit = (np.random.default_rng(0).random(len(pr)) - 0.5) * 0.3
        ax.scatter(pr + jit, me + (0.05 if mk == "x" else -0.05), marker=mk, label=lab, alpha=0.6)
    ax.plot([1, 4], [1, 4], "k--", lw=1, label="equation (11)")
    ax.set_xlabel("predicted width, eq. (11)")
    ax.set_ylabel("widest global load measured")
    ax.set_xticks([1, 2, 4])
    ax.set_yticks([1, 2, 4])
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "e5-width.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
