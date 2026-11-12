"""Size scan of the multi-spin-coded mode against the specialised 2D Ising codes (results figure); usage: plot_rd_ising_scan.py <e6-scan.csv> <out.pdf>."""
import csv
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("pdf")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
panels = [("T4 (v12, warm-up)", "T4"), ("A100-80GB", "A100-80GB"), ("H100", "H100")]
plt.rcParams.update({
    "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 7.6, "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4, "legend.fontsize": 7.4, "font.family": "serif", "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
})
fig, axes = plt.subplots(1, 3, figsize=(4.72, 1.95), constrained_layout=True)
series = [("Ours r10", "generated, r10", "#20252B", "o", "-"),
          ("Ours r7", "generated, r7", "#7D848C", "o", "--"),
          ("ising-gpu", "ising-gpu", "#4F7CAC", "s", "-"),
          ("cuIsing 2025", "cuIsing 2025", "#E09F3E", "^", "-")]
for ax, (key, title) in zip(axes, panels):
    sel = [r for r in rows if r["Device"] == key]
    x = [int(r["Side"]) for r in sel]
    for col, label, color, marker, ls in series:
        ax.plot(x, [float(r[col]) for r in sel], ls, color=color, marker=marker, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v // 1024}k" for v in x])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title, pad=2)
    ax.set_xlabel("lattice side")
    ax.tick_params(length=2, pad=1.5)
axes[0].set_ylabel("spin updates per ns")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8,
           columnspacing=0.8)
fig.savefig(out)
