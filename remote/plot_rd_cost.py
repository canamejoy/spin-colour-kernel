"""Cost figure for the results section: rate per system at B = 128 on the H100 with the nanodisk application; usage: plot_rd_cost.py <e6-throughput-h100.csv> <e9-nanodisk-q.csv> <out.pdf>."""
import csv
import statistics
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

throughput, nanodisk, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(throughput)))
names = {"Heisenberg nanodot": "Heisenberg nanodot, masked"}
labels = [names.get(r["System"], r["System"]) for r in rows]
rates = [float(r["Flips/ns B=128"]) for r in rows]
disk = {}
for r in csv.DictReader(open(nanodisk)):
    disk.setdefault(r["Colours"], []).append(float(r["flips/ns"]))
disk = {k: statistics.median(v) for k, v in disk.items()}
labels += ["nanodisk, shell 1 (2 colours)", "nanodisk, shells 1, 2, 4 (7 colours)"]
rates += [disk["2"], disk["7"]]

plt.rcParams.update({"font.size": 7.6, "axes.labelsize": 7.6, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
                     "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(4.72, 2.9), constrained_layout=True)
y = list(range(len(labels)))[::-1]
colors = ["#4F7CAC"] * len(rows) + ["#7D848C", "#E09F3E"]
ax.barh(y, rates, color=colors, height=0.7)
for yi, v in zip(y, rates):
    ax.text(v + 1.0, yi, f"{v:.2f}", va="center", fontsize=7.4)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("spin updates per ns")
ax.set_xlim(0, max(rates) * 1.15)
ax.tick_params(length=2, pad=1.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.savefig(out)
