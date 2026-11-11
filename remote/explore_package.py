"""Package the exploration runs into a folder ready for a Kaggle dataset.

    python remote/explore_package.py --runs results/explore/stage1 ... --an results/explore/all \
        --out results/explore/kaggle

Writes: atlas.png (eight examples per class), the per-class galleries and the phase map from the
analysis folder, states.csv (one row per state: parameters, majority class and class fractions over
replicas, mean Q and Mz), and textures.npz (final middle-layer s_z of every replica as int8 = round
(127 s_z), with params, names, run and the majority class).
"""
import argparse
import csv
import glob
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PAR = ("Jex2", "Jex3", "Jex4", "Kan1", "KanS", "Hex", "KDM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--an", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "galleries").mkdir(parents=True, exist_ok=True)
    an = Path(a.an)
    for p in an.glob("gallery_*.png"):
        shutil.copy(p, out / "galleries" / p.name.replace("gallery_", ""))
    shutil.copy(an / "phase_map.png", out / "phase_map.png")

    lab = defaultdict(Counter)
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(an / "descriptors.csv")):
        k = (r["run"], r["name"])
        lab[k][r["label"]] += 1
        for f in ("Q", "mz"):
            acc[k][f].append(float(r[f]))
    classes = sorted({c for v in lab.values() for c in v})

    sz, params, names, runs, major = [], [], [], [], []
    for run in a.runs:
        for f in sorted(glob.glob(f"{run}/chunk_*.npz")):
            z = np.load(f)
            for i, nm in enumerate(z["names"]):
                k = (Path(run).name, str(nm))
                sz.append(np.round(127 * z["final"][i, :, 2].astype(np.float32)).astype(np.int8))
                params.append(z["params"][i])
                names.append(str(nm))
                runs.append(k[0])
                major.append(lab[k].most_common(1)[0][0])
    np.savez_compressed(out / "textures.npz", sz_final_int8=np.stack(sz), params=np.stack(params),
                        param_names=np.asarray(PAR), names=np.asarray(names), run=np.asarray(runs),
                        majority_class=np.asarray(major))
    with open(out / "states.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "name", *PAR, "majority_class", *[f"frac_{c}" for c in classes], "Q_mean", "Mz_mean"])
        for nm, rn, p, mj in zip(names, runs, params, major):
            k = (rn, nm)
            n = sum(lab[k].values())
            w.writerow([rn, nm, *[f"{x:.4f}" for x in p], mj, *[f"{lab[k][c] / n:.3f}" for c in classes],
                        f"{np.mean(acc[k]['Q']):.3f}", f"{np.mean(acc[k]['mz']):.3f}"])

    # atlas: eight states per class, replica 0, spread over the class's states
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    N = sz[0].shape[-1]
    c = N // 2
    y, x = np.mgrid[:N, :N] - c
    mask = (x ** 2 + y ** 2) <= 18.30 ** 2
    fig, axs = plt.subplots(len(classes), 8, figsize=(12, 1.6 * len(classes)))
    for i, cl in enumerate(classes):
        idx = [j for j, m in enumerate(major) if m == cl]
        pick = rng.choice(idx, min(8, len(idx)), replace=False) if idx else []
        for j in range(8):
            ax = axs[i, j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(np.where(mask, sz[pick[j]][0] / 127, np.nan), cmap="RdBu_r", vmin=-1, vmax=1,
                          origin="lower", interpolation="nearest")
                p = params[pick[j]]
                ax.set_title(f"D{p[6]:.2f} H{p[5]:.2f} J2{p[0]:+.2f}", fontsize=6)
        axs[i, 0].text(-0.15, 0.5, f"{cl}\n({len(idx)})", transform=axs[i, 0].transAxes, ha="right",
                       va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "atlas.png", dpi=120)
    plt.close(fig)
    print(len(names), "states;", Counter(major).most_common())


if __name__ == "__main__":
    main()
