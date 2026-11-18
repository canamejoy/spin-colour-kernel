"""Bootstrap p value of each E4 critical temperature against its primary-source reference.

usage: e4_reference_p.py [--device cuda] <run-dir>=<primary reference> [...]
E4 computed p against the rounded reference it ran with (system.TC). This repeats E4's own bootstrap -- the same
estimate (e4_jackknife.estimator), 2000 seed resamples drawn from numpy default_rng(1) -- so the interval it
reproduces can be checked against the run's own row, and evaluates the two-sided bootstrap p value of the
primary-source reference, Holm-adjusted across the runs given.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e4_jackknife import estimator  # noqa: E402
from Spin_Colour_Kernel.exp_core import holm  # noqa: E402


def reference_p(run: Path, ref: float, boots: int = 2000, device: str = "cpu") -> dict:
    row, estimate, seeds, _ = estimator(run, device=device)
    est = estimate(None)
    rng = np.random.default_rng(1)
    bs = np.array([estimate(np.bincount(rng.integers(0, seeds, seeds), minlength=seeds).astype(np.float64))
                   for _ in range(boots)])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"system": row["system"], "estimate": est, "low": float(lo), "high": float(hi),
            "row_low": row["low"], "row_high": row["high"], "run_reference": row["T_ref"], "reference": ref,
            "p_run_reference": float(min(1.0, 2 * min(np.mean(bs <= row["T_ref"]), np.mean(bs >= row["T_ref"])))),
            "p_row": row["p_reference"],
            "p": float(min(1.0, 2 * min(np.mean(bs <= ref), np.mean(bs >= ref)))), "boots": int(len(bs))}


if __name__ == "__main__":
    args = sys.argv[1:]
    device = "cpu"
    if "--device" in args:
        i = args.index("--device"); device = args[i + 1]; del args[i:i + 2]
    out = [reference_p(Path(a.split("=")[0]), float(a.split("=")[1]), device=device) for a in args]
    for r, pa in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = pa
    print(json.dumps(out, indent=1))
