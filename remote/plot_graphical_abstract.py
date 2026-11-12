"""Nanodisk panels of the graphical abstract, redrawn from the raw E9 states with large type.

usage: plot_graphical_abstract.py <dir-shell-1> <dir-shells-124> [<dir-shells-124-b> ...] <out.pdf>
Each directory holds state_E9s*-H<field>.npz files; obs[:, 5, :] is the topological charge Q per
temperature and replica.
"""
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("pdf")
import matplotlib.pyplot as plt

*dirs, out = sys.argv[1:]
runs = {"1": {}, "124": {}}
for d in dirs:
    for f in glob.glob(f"{d}/state_E9s*-H*.npz"):
        m = re.search(r"E9s(1|124)-H([0-9.]+)\.npz$", f)
        z = np.load(f)
        runs[m.group(1)][float(m.group(2))] = (z["temperatures"], z["obs"][:, 5, :].mean(axis=1))

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.7,
                     "lines.linewidth": 1.5})
fig, axes = plt.subplots(3, 1, figsize=(3.6, 5.6), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for ax, key, title in ((axes[0], "1", "exchange shell 1 (2 colours)"),
                       (axes[1], "124", "shells 1, 2 and 4 (7 colours)")):
    fields = sorted(runs[key])
    for k, h in enumerate(fields):
        T, Q = runs[key][h]
        ax.plot(T, Q, color=cmap(k / max(1, len(fields) - 1)), label=f"H = {h:.1f}")
    ax.set_xscale("log")
    ax.set_title(title, pad=3)
    ax.set_ylabel("Q")
    ax.set_xlabel("T (K)")
axes[0].legend(ncol=2, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.47), handlelength=1.4, columnspacing=0.8)
for key, label, color, marker in (("1", "shell 1", "#4F7CAC", "o"), ("124", "shells 1, 2, 4", "#E09F3E", "s")):
    fields = sorted(runs[key])
    axes[2].plot(fields, [runs[key][h][1][-1] for h in fields], marker=marker, color=color, label=label)
axes[2].set_xlabel("field H")
axes[2].set_ylabel("Q at 0.1 K")
axes[2].legend(frameon=False, loc="lower right")
for ax in axes:
    ax.tick_params(length=2.5, pad=2)
fig.savefig(out)
