"""Fold later confirmatory runs into Spin-Colour-Kernel/Results/significance.json and fix the E2 effect size.

Idempotent: entries it owns (comparison prefixes in OWNED) are rebuilt on every run.

- E2/E2b: the effect size is Cohen's w = sqrt(chi2 / n). Cramer's V = sqrt(chi2 / (n (k - 1))) is bounded by 1
  only for contingency tables; for a goodness-of-fit test against bins with tiny expected counts (the all-sites
  control) the normalised value exceeds 1 (16.5), so it was the wrong name and the wrong bound.
- H5(b): the registered verdict uses the default 10 Philox rounds (proposal r15, experiments v12); the 7-round
  value is kept as an ablation.
- E2 confirmatory seeds, pooled total-variation distances, and the E4, E4tri v07, E5conf, E6conf and E9 verdict
  rows (from the product tables that hold them).
- E6scan v12 (T4, warm-up, clocks), E6vamp v12 (32^3-64^3, warm-up) and v13 (H100, 64^3-96^3, 10 rounds) per row.

    python remote/fold_significance.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "Spin-Colour-Kernel" / "Results" / "significance.json"
R = ROOT / "results"
SCAN_V12 = {"T4": R / "t4/20261110-150849/E6scan-v12.json"}
VAMP_V12 = {"H100": R / "h100/20261110-150850/E6vamp-v12.json", "A100-80GB": R / "a100/20261110-150849/E6vamp-v12.json"}
VAMP_V13 = {"H100": R / "h100/20261110-164945/E6vamp-v13.json"}
OWNED = ("E6scan v12 ", "E6vamp v12 ", "E6vamp v13 ", "H5(b) verdict ", "E2conf ", "E2 pooled TV ", "E4 Tc ",
         "E4tri v07 ", "E5conf ", "E6conf ", "E9 ")
K = R / "kaggle"
E2_RUNS = {"E2": K / "e/results/E2.json", "E2b": K / "f/results/E2b.json", "E2conf": K / "e2conf/results/E2conf.json",
           "E2conf-metropolis": K / "e2conf/results/E2conf-metropolis.json"}
TABLES = ROOT / "Spin-Colour-Kernel" / "Results" / "tables"


def table(name):
    import csv
    with (TABLES / name).open() as f:
        return list(csv.DictReader(f))


def total_variation(h):
    """Half the L1 distance between the pooled sampled and the exact energy distributions, in [0, 1]."""
    return 0.5 * sum(abs(a - b) for a, b in zip(h["sampled"], h["exact"]))


def folded_from_runs_and_tables():
    out = []
    for tag, p in E2_RUNS.items():
        d = json.loads(p.read_text())
        n, acc = d["params"]["samples"], d["params"].get("acceptance", "metropolis")
        for r in d["rows"]:
            if tag.startswith("E2conf"):
                for s in r["seeds"]:
                    out.append({"comparison": f"E2conf {r['lattice']} {r['rule']} ({acc}) seed {s['seed']}",
                                "test": "chi-square goodness of fit against exact enumeration", "statistic": s["chi2"],
                                "p_value": s["p"], "p_adjusted": s["p_holm"], "correction": "Holm over 10 seeds",
                                "effect_size": math.sqrt(s["chi2"] / n),
                                "effect_size_name": "Cohen's w (goodness of fit, sqrt(chi2/n))", "n": n, "alpha": 0.01,
                                "hypothesis": "H1b", "experiment": "E2", "note": "confirmatory (v05), seeds 1001-1010"})
            h = d["histograms"][f"{r['lattice']}|{r['rule']}"]
            tv = total_variation(h)
            out.append({"comparison": f"E2 pooled TV {tag} {r['lattice']} {r['rule']} ({acc})",
                        "test": "total-variation distance, pooled sampled vs exact energy distribution",
                        "statistic": tv, "p_value": None, "p_adjusted": None, "correction": "none (descriptive)",
                        "effect_size": tv, "effect_size_name": "total-variation distance (0 = identical, 1 = disjoint)",
                        "n": n * len(r["seeds"]), "alpha": None, "hypothesis": "H1b", "experiment": "E2",
                        "note": f"{tag}, {len(r['seeds'])} seeds pooled"})
    for r in table("e4-critical-points-h100.csv"):
        out.append({"comparison": f"E4 Tc {r['System']} (H100)", "test": "reference inside the bootstrap interval of Tc",
                    "statistic": float(r["Tc"]), "p_value": float(r["p"]), "p_adjusted": float(r["p (Holm)"]),
                    "correction": "Holm over 8 systems", "effect_size": float(r["Half width / Tc (%)"]),
                    "effect_size_name": "interval half width / Tc (%)", "n": None, "alpha": 0.05, "hypothesis": "H3",
                    "experiment": "E4", "interval": [float(r["Interval low"]), float(r["Interval high"])],
                    "note": f"reference {r['Reference']}; {r['Verdict']}; source table e4-critical-points-h100.csv"})
    for r in table("e4-triangular-kpoint-v07.csv"):
        if not r.get("High"):        # the table's second block holds the finite-size slopes
            break
        lr = {"mean": float(r["Log ratio"]), "low": float(r["Low"]), "high": float(r["High"])}
        e = paired(f"E4tri v07 T={r['T']} L {r['L1']}->{r['L2']}", lr, None, "K-point structure factor; "
                   "source table e4-triangular-kpoint-v07.csv", hyp="H3")
        e.update({"test": "bootstrap of the log ratio of K-point structure factors", "experiment": "E4",
                  "effect_size_name": "log ratio"})
        out.append(e)
    for r in table("e5-conf-unroll-cost.csv"):
        lr = {"mean": float(r["Pooled log ratio"]), "low": float(r["Interval low"]), "high": float(r["Interval high"])}
        e = paired(f"E5conf {r['Device']} pooled u=4 vs odd neighbours", lr, None,
                   f"(a) {r['(a) grids holding']}; |pooled| >= log 1.3: {r['|pooled| >= log 1.3']}; "
                   "source table e5-conf-unroll-cost.csv", hyp="H4")
        e.update({"test": "paired bootstrap of the pooled log time ratio", "experiment": "E5"})
        out.append(e)
    for r in table("e6-conf-slopes.csv"):
        out.append({"comparison": f"E6conf batch slope {r['Device']} {r['System']}",
                    "test": "bootstrap of the log-cost slope per batch doubling, one-sided equivalence bound",
                    "statistic": float(r["Log-cost slope"]), "p_value": None, "p_adjusted": None,
                    "correction": "none (interval against the log 1.05 margin)", "effect_size": float(r["Log-cost slope"]),
                    "effect_size_name": "log-cost slope per doubling", "n": 10, "alpha": 0.05, "hypothesis": "H5",
                    "experiment": "E6", "interval": [float(r["Slope low"]), float(r["Slope high"])],
                    "note": f"below log 1.05: {r['Below log 1.05']}; role {r['Role']}; source table e6-conf-slopes.csv"})
    for r in table("e9-nanodisk-q.csv"):
        hw = float(r["Max bootstrap half-width over T"])
        out.append({"comparison": f"E9 Q shells {r['Shell set']} H={r['Field H']}",
                    "test": "bootstrap half-width of Q against the 0.1 precision bound", "statistic": float(r["Q at 0.1 K"]),
                    "p_value": None, "p_adjusted": None, "correction": "none (bound check)", "effect_size": hw,
                    "effect_size_name": "max bootstrap half-width of Q over T", "n": None, "alpha": 0.05,
                    "hypothesis": "H6", "experiment": "E9", "interval": [float(r["Low"]), float(r["High"])],
                    "note": f"within 0.1: {hw <= 0.1}; source table e9-nanodisk-q.csv"})
    return out
MARGIN = -math.log(1.05)


def paired(comparison, lr, n, note, hyp="H5"):
    return {"comparison": comparison, "test": "paired bootstrap of the log speed ratio, 95% percentile interval",
            "statistic": lr["mean"], "p_value": None, "p_adjusted": None, "correction": "none (interval reported)",
            "effect_size": lr["mean"], "effect_size_name": "log speed ratio", "n": n, "alpha": 0.05,
            "hypothesis": hyp, "experiment": "E6", "interval": [lr["low"], lr["high"]], "note": note}


# papersmith reads at most 200 rows per significance*.json file, so the record is split by experiment family
SPLIT = (("significance-e6vamp.json", ("E6vamp ",)), ("significance-e2conf.json", ("E2conf ", "E2 pooled TV ")),
         ("significance-e6scan.json", ("E6scan ", "H5(b) verdict ", "E6bitind ")),
         ("significance-e4-e5-e6conf-e9.json", ("E4 Tc ", "E4tri v07 ", "E5conf ", "E6conf ", "E9 ")))
MAX_ROWS = 200


def write_split(tests):
    seen = {}
    for t in tests:                                   # keep the last entry of a repeated comparison
        seen[t["comparison"]] = t
    tests = list(seen.values())
    groups = {name: [] for name, _ in SPLIT}
    rest = []
    for t in tests:
        name = next((n for n, pre in SPLIT if t["comparison"].startswith(pre)), None)
        (groups[name] if name else rest).append(t)
    for name, rows in [("significance.json", rest)] + list(groups.items()):
        assert len(rows) <= MAX_ROWS, (name, len(rows))
        (SIG.parent / name).write_text(json.dumps({"tests": rows}, indent=2))
        print(name, len(rows))


def main():
    tests = []
    for f in sorted(p for p in SIG.parent.glob("significance*.json") if "v14-v16" not in p.name and "v17" not in p.name):  # merge earlier split
        tests += json.loads(f.read_text())["tests"]
    tests = [t for t in tests if not t["comparison"].startswith(OWNED)]
    for t in tests:
        if t.get("experiment") in ("E2", "E2b") and t.get("statistic") is not None:
            t["effect_size"] = math.sqrt(t["statistic"] / t["n"])
            t["effect_size_name"] = "Cohen's w (goodness of fit, sqrt(chi2/n))"
        if "H5(b) verdict" in t["comparison"] and t["comparison"].startswith("E6scan ") \
                and "ablation" not in t.get("note", ""):
            t["note"] = t.get("note", "") + "; 7 Philox rounds, ablation (the registered verdict uses 10 rounds)"
    by = {t["comparison"]: t for t in tests}
    for dev in ("H100", "A100-80GB"):
        k = f"E6scan {dev} 32768^2 r10_vs_cuIsing-2025"
        t = by[k]
        lr = {"mean": t["statistic"], "low": t["interval"][0], "high": t["interval"][1]}
        tests.append(paired(f"H5(b) verdict {dev} (cuIsing-2025 at 32768^2, 10 rounds)", lr, t["n"],
                            f"v11 run, verdict at the v12 default; non-inferior={lr['low'] > MARGIN} margin {MARGIN:.4f}"))
    for dev, p in SCAN_V12.items():
        d = json.loads(p.read_text())
        for row in d["rows"]:
            for key, lr in row.items():
                if key.startswith(("r7_vs", "r10_vs")):
                    tests.append(paired(f"E6scan v12 {dev} {row['side']}^2 {key}", lr, 10,
                                        f"v12 rerun, {row.get('warmup_rounds_discarded', 0)} warm-up rounds discarded"))
        v = d["verdict"]
        tests.append(paired(f"H5(b) verdict {dev} ({v['specialist']} at {v['side']}^2, {v['verdict_rounds']} rounds)",
                            v["log_ratio"], 10, f"v12; non-inferior={v['non_inferior']} margin {v['margin']:.4f}"))
    for tag, dev, p in [("v12", k, v) for k, v in VAMP_V12.items()] + [("v13", k, v) for k, v in VAMP_V13.items()]:
        d = json.loads(p.read_text())
        for row in d["rows"]:
            for b in ("1", "1d", "128"):
                lr = row.get(f"log_ratio_B{b}")
                if lr:
                    tests.append(paired(f"E6vamp {tag} {dev} {row['system']} {row['cells']}^3 B={b}", lr, d["reps"],
                                        "VAMPIRE double precision; B=1d is ours in float64; "
                                        f"{d['warmup_rounds_discarded']} warm-up round discarded"))
    tests += folded_from_runs_and_tables()
    write_split(tests)
    print(len(tests), "tests")


if __name__ == "__main__":
    main()
