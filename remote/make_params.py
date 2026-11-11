"""Draw exploration parameter rows: a scrambled Sobol design over a box, with a shell mode per row."""
import json, sys
import numpy as np
from scipy.stats import qmc

BOX = {"Jex2": (-0.4, 0.7), "Jex3": (-0.3, 0.3), "Jex4": (-0.3, 0.3), "Kan1": (-0.3, 0.8),
       "KanS": (-0.4, 0.4), "Hex": (0.0, 1.6), "KDM": (0.0, 1.8)}


def draw(n, seed, box=BOX, frac=(0.6, 0.3, 0.1)):
    lo = np.array([b[0] for b in box.values()]); hi = np.array([b[1] for b in box.values()])
    X = qmc.scale(qmc.Sobol(7, scramble=True, seed=seed).random(n), lo, hi)
    mode = np.random.default_rng(seed).choice(3, n, p=frac)     # 0: shell 1, 1: +Jex2, 2: +Jex2,3,4
    X[mode == 0, 0:3] = 0.0
    X[mode == 1, 1:3] = 0.0
    return np.round(X, 4), mode


if __name__ == "__main__":
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    fixed = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {"names": [], "params": []}
    X, mode = draw(n, seed)
    names = fixed["names"] + [f"s{seed}-{i:04d}-m{m}" for i, m in enumerate(mode)]
    rows = fixed["params"] + X.tolist()
    json.dump({"names": names, "params": rows, "box": BOX}, open(out, "w"), indent=0)
    print(len(rows), "rows ->", out)


def draw_reduced(n, seed, frac=(0.45, 0.5, 0.05)):
    """Stage 2: sample the field as h = Hex / H_c with the stage-1 saturation boundary
    H_c = 0.62 KDM^2 / (1 + 2.5 Jex2), so few states land in the saturated phase."""
    U = qmc.Sobol(7, scramble=True, seed=seed).random(n)
    mode = np.random.default_rng(seed).choice(3, n, p=frac)
    J2 = np.where(mode > 0, -0.15 + 0.85 * U[:, 0], 0.0)
    J3 = np.where(mode == 2, -0.2 + 0.4 * U[:, 1], 0.0)
    J4 = np.where(mode == 2, -0.2 + 0.4 * U[:, 2], 0.0)
    K1 = -0.3 + 1.1 * U[:, 3]
    KS = -0.4 + 0.8 * U[:, 4]
    D = 0.05 + 1.75 * U[:, 6]
    Hc = 0.62 * D ** 2 / (1 + 2.5 * J2)
    H = np.minimum(1.25 * U[:, 5] * Hc, 1.6)
    return np.round(np.stack([J2, J3, J4, K1, KS, H, D], 1), 4), mode


def draw_lowfield(n, seed, frac=(0.3, 0.7)):
    """Stage 3: the low-field corner h = Hex / H_c <= 0.45, where one to a few large domains form."""
    U = qmc.Sobol(6, scramble=True, seed=seed).random(n)
    mode = np.random.default_rng(seed).choice(2, n, p=frac)
    J2 = np.where(mode == 1, 0.7 * U[:, 0], 0.0)
    K1 = -0.3 + 1.1 * U[:, 1]
    KS = -0.4 + 0.8 * U[:, 2]
    D = 0.05 + 1.35 * U[:, 4]
    H = 0.45 * U[:, 3] * 0.62 * D ** 2 / (1 + 2.5 * J2)
    Z = np.zeros(n)
    return np.round(np.stack([J2, Z, Z, K1, KS, H, D], 1), 4), mode
