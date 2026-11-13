"""Leave-one-seed-out jackknife of the E4 critical-temperature estimate, from the saved raw series.

usage: e4_jackknife.py <run-dir> [<run-dir> ...]
--drop-smallest refits without the smallest lattice, a check on corrections to scaling.
Each run directory holds the <tag>-<system>-L<L>.npz series E4 saved (E, m2, m4 of shape (T, seed, t)) and
its <tag>-<system>.json row. The estimate is recomputed with the same multi-histogram Binder curves, crossing
grids and size extrapolation E4 uses (exp_perf._binder_curve, _crossing), once with every seed and once with
each seed left out; the jackknife standard error gives a t interval with seeds - 1 degrees of freedom.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Spin_Colour_Kernel import exp_perf as P  # noqa: E402

T975 = {9: 2.262157}   # Student t quantile, 9 degrees of freedom


def jackknife(run: Path, drop_smallest: bool = False) -> dict:
    row = json.loads(next(run.glob("E4*.json")).read_text())["rows"][0]
    name, x, Tc = row["system"], row["correction_exponent"], row["T_ref"]
    data = {}
    for f in sorted(run.glob("*-L*.npz")):
        z = np.load(f)
        data[int(f.stem.rsplit("-L", 1)[1])] = {k: z[k] for k in ("E", "m2", "m4", "temps")}
    Ls = sorted(data)
    if drop_smallest:
        Ls = Ls[1:]
    pairs = list(zip(Ls[:-1], Ls[1:]))
    grids = {(a, b): np.linspace(max(data[a]["temps"][0], data[b]["temps"][0]),
                                 min(data[a]["temps"][-1], data[b]["temps"][-1]), 241) for a, b in pairs}
    lnZ_full = {}

    def estimate(w):
        cur = {}
        for (a, b), g in grids.items():
            for L in (a, b):
                d = data[L]
                ww = None if w is None else np.broadcast_to(w[None, :, None], d["E"].shape)
                u, lnZ = P._binder_curve(d["E"], d["m2"], d["m4"], d["temps"], g, weights=ww, device="cpu",
                                         lnZ0=lnZ_full.get(L))
                if w is None:
                    lnZ_full[L] = lnZ
                cur[(a, b, L)] = u
        cr = [(math.sqrt(a * b), P._crossing(grids[(a, b)], cur[(a, b, a)], cur[(a, b, b)])) for a, b in pairs]
        cr = [(l, t) for l, t in cr if np.isfinite(t)]
        X = np.array([l ** (-x) for l, _ in cr])
        Y = np.array([t for _, t in cr])
        return float(np.polyfit(X, Y, 1)[1])

    full = estimate(None)
    seeds = data[Ls[0]]["E"].shape[1]
    loo = []
    for s in range(seeds):
        w = np.ones(seeds)
        w[s] = 0.0
        loo.append(estimate(w))
    loo = np.array(loo)
    se = math.sqrt((seeds - 1) / seeds * np.sum((loo - loo.mean()) ** 2))
    t = T975[seeds - 1]
    return {"system": name, "sizes": Ls, "estimate": full, "bootstrap_estimate": row["estimate"], "reference": Tc,
            "jackknife_se": se, "t_low": full - t * se, "t_high": full + t * se,
            "t_contains_reference": bool(full - t * se <= Tc <= full + t * se),
            "bootstrap_low": row["low"], "bootstrap_high": row["high"],
            "t_half_over_Tc": t * se / Tc, "bootstrap_half_over_Tc": row["half_width_over_Tc"],
            "loo": loo.tolist()}


if __name__ == "__main__":
    drop = "--drop-smallest" in sys.argv
    out = [jackknife(Path(d), drop) for d in sys.argv[1:] if d != "--drop-smallest"]
    print(json.dumps(out, indent=1))
