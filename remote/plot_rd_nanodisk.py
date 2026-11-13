"""Nanodisk figure for the results section, redrawn from the raw E9 states.

usage: plot_rd_nanodisk.py <dir> [<dir> ...] <out.png>
Each directory holds state_E9s*-H<field>.npz files; obs[:, 5, :] is the skyrmion number N_sk per temperature
and replica. Left and middle: N_sk against temperature per field for shell 1 and for shells 1, 2 and 4; right:
N_sk at the lowest temperature against the field.
"""
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

*dirs, out = sys.argv[1:]
runs = {"1": {}, "124": {}}
for d in dirs:
    for f in glob.glob(f"{d}/state_E9s*-H*.npz"):
        m = re.search(r"E9s(1|124)-H([0-9.]+)\.npz$", f)
        z = np.load(f)
        runs[m.group(1)][float(m.group(2))] = (z["temperatures"], z["obs"][:, 5, :].mean(axis=1))

plt.rcParams.update({"font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 7.6, "xtick.labelsize": 7.4,
                     "ytick.labelsize": 7.4, "legend.fontsize": 7.0, "font.family": "serif",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1})
fig, axes = plt.subplots(1, 3, figsize=(4.72, 1.95), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for ax, key, title in ((axes[0], "1", "shell 1 (2 colours)"), (axes[1], "124", "shells 1, 2, 4 (7 colours)")):
    fields = sorted(runs[key])
    for k, h in enumerate(fields):
        T, q = runs[key][h]
        ax.plot(T, q, color=cmap(k / max(1, len(fields) - 1)), label=f"H = {h:.1f}")
    ax.set_xscale("log")
    ax.set_title(title, pad=2)
    ax.set_xlabel("T (K)")
axes[0].set_ylabel(r"$N_{\mathrm{sk}}$")
axes[0].legend(ncol=1, frameon=False, loc="lower left", handlelength=1.2, labelspacing=0.2, fontsize=6.0)
for key, label, color, marker in (("1", "1", "#4F7CAC", "o"), ("124", "1, 2, 4", "#E09F3E", "s")):
    fields = sorted(runs[key])
    axes[2].plot(fields, [runs[key][h][1][-1] for h in fields], marker=marker, markersize=3, color=color, label=label)
axes[2].set_xlabel("field H")
axes[2].set_ylabel(r"$N_{\mathrm{sk}}$ at 0.1 K")
axes[2].legend(frameon=False, loc="lower right", handlelength=1.0, fontsize=6.0, borderaxespad=0.2, title="shells", title_fontsize=6.0)
for ax in axes:
    ax.tick_params(length=2, pad=1.5)
fig.savefig(out, dpi=300)
