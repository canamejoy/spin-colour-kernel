"""Texture descriptors and galleries for the exploration runs (results/explore/<run>/chunk_*.npz).

    python remote/explore_analyze.py results/explore/<run> [more runs...] --out results/explore/<run>/an

Per replica, on the final middle layer inside the disk:
  mz, m_ip       mean s_z and in-plane magnetization norm
  Q              Berg-Luscher charge (from the run)
  sz_sd          spread of s_z
  n_dom          domains of either sign (|s_z| > 0.3, 4-connected, >= 3 sites)
  n_bub          minority-sign bubbles: closed (not touching the rim), compact, area <= 120
  q_peak, aniso  peak wavenumber of the s_z structure factor and its angular anisotropy (stripes high)
  wind           winding of the in-plane magnetization along a circle at r = Rd - 2 (vortex = +-1)
and a rule-based label per replica. The labels are a first sorting for the galleries, never ground
truth: every class is checked by eye on the galleries.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

RD = 18.30


def disk_mask(N):
    c = N // 2
    y, x = np.mgrid[:N, :N] - c
    return (x ** 2 + y ** 2) <= RD ** 2


def winding(sx, sy, N, r):
    c = N // 2
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    xi = np.clip(np.round(c + r * np.cos(th)).astype(int), 0, N - 1)
    yi = np.clip(np.round(c + r * np.sin(th)).astype(int), 0, N - 1)
    phi = np.arctan2(sy[yi, xi], sx[yi, xi])
    d = np.diff(np.concatenate([phi, phi[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d.sum() / (2 * np.pi)


def descriptors(f, Q, mask):
    sx, sy, sz = (f[i].astype(np.float64) for i in range(3))
    N = sz.shape[0]
    z = sz[mask]
    mz = z.mean()
    m_ip = np.hypot(sx[mask].mean(), sy[mask].mean())
    # domains
    lab_p, npos = ndimage.label((sz > 0.3) & mask)
    lab_n, nneg = ndimage.label((sz < -0.3) & mask)
    rim = mask & ~ndimage.binary_erosion(mask)
    minority = lab_n if mz >= 0 else lab_p
    n_min = nneg if mz >= 0 else npos
    sizes = lambda lab, n: ndimage.sum(np.ones_like(sz), lab, range(1, n + 1)) if n else np.zeros(0)
    n_dom = int((sizes(lab_p, npos) >= 3).sum() + (sizes(lab_n, nneg) >= 3).sum())
    n_bub, n_giant, bub_area = 0, 0, []
    for k, sl in enumerate(ndimage.find_objects(minority), start=1):
        if sl is None:
            continue
        reg = minority[sl] == k
        a = reg.sum()
        if a < 3 or (rim[sl] & reg).any():
            continue
        if a > 120:
            n_giant += 1
            continue
        h, w = reg.shape
        if max(h, w) / max(1, min(h, w)) <= 2.2 and a / (h * w) >= 0.45:
            n_bub += 1
            bub_area.append(a)
    # structure factor of s_z
    g = np.where(mask, sz - z.mean(), 0.0)
    P = np.abs(np.fft.fftshift(np.fft.fft2(g, s=(64, 64)))) ** 2
    ky, kx = np.mgrid[-32:32, -32:32]
    kr = np.hypot(kx, ky)
    rb = np.bincount(kr.astype(int).ravel(), P.ravel(), minlength=46)[:46]
    rb[:2] = 0
    kp = int(np.argmax(rb))
    q_peak = 2 * np.pi * kp / 64
    ring = (np.abs(kr - kp) <= 1.0) & (kr >= 2)
    ang = np.arctan2(ky[ring], kx[ring]) % np.pi
    ab = np.bincount((ang / np.pi * 12).astype(int).clip(0, 11), P[ring], minlength=12)
    aniso = float(ab.max() / (ab.mean() + 1e-12))
    wind = winding(sx, sy, N, RD - 2.0)
    return dict(mz=mz, m_ip=m_ip, Q=float(Q), sz_sd=z.std(), n_dom=n_dom, n_bub=n_bub, n_giant=n_giant,
                bub_area=float(np.mean(bub_area)) if bub_area else 0.0,
                q_peak=q_peak, aniso=aniso, wind=float(np.round(wind, 2)))


def label(d):
    if d["n_dom"] >= 3 and d["q_peak"] > 2 * np.pi / 4.5:
        return "lattice-scale"          # period <= 4.5 sites: modulation at the pixel scale of the image
    if d["n_bub"] >= 3 and d["bub_area"] < 3:
        return "lattice-scale"          # skyrmions of one or two sites
    if abs(d["mz"]) > 0.85 and d["sz_sd"] < 0.15:
        return "saturated-z"
    if d["m_ip"] > 0.75 and d["sz_sd"] < 0.3:
        return "inplane-FM"
    if d["n_giant"] >= 1 and d["n_dom"] <= 3:
        return "giant-bubble"           # one closed reversed domain larger than 120 sites (target-like)
    if abs(d["wind"]) > 0.8 and d["n_dom"] <= 2 and abs(d["mz"]) < 0.5:
        return "hook-or-wall"     # in-plane winding at the rim: hooked single stripe, domain wall or target
    if d["n_bub"] >= 1 and abs(d["Q"]) >= 0.6:
        if d["n_bub"] <= 3 and d["n_bub"] >= d["n_dom"] - 1:
            return "isolated-skyrmions"  # one to three closed skyrmions in a uniform background
        majority_closed = d["n_bub"] >= max(2, 0.6 * (d["n_dom"] - 1))
        if d["bub_area"] < 8:
            return "small-skyrmions"    # cores of 3-8 sites: real, but at the image's resolution limit
        return "skyrmion-lattice" if d["n_bub"] >= 6 and majority_closed else (
            "skyrmions" if majority_closed else "stripes+skyrmions")
    if d["n_dom"] <= 2 and d["sz_sd"] > 0.35:
        return "two-domain"             # two out-of-plane domains split by one wall
    if 3 <= d["n_dom"] <= 4 and d["sz_sd"] > 0.35:
        return "few-domain"             # a single stripe or meander across the disk
    if d["n_dom"] >= 3 and d["aniso"] >= 3.0:
        return "helical-stripes"
    if d["n_dom"] >= 3:
        return "labyrinth"
    if d["sz_sd"] < 0.3:
        return "canted-uniform"
    return "other"


def load(runs):
    data = []
    for r in runs:
        for f in sorted(glob.glob(f"{r}/chunk_*.npz")):
            z = np.load(f, allow_pickle=False)
            data.append(dict(run=Path(r).name, **{k: z[k] for k in z.files}))
    return data


def gallery(items, path, title, ncol=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    if not n:
        return
    nrow = -(-n // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.6 * ncol, 1.85 * nrow + 0.4), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, (img, cap) in zip(axs.ravel(), items):
        a.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        a.set_title(cap, fontsize=5.2)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def phase_map(rows, out):
    """Majority label per state on (KDM, Hex), one panel per shell mode, coloured by label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    st = defaultdict(Counter)
    par = {}
    for r in rows:
        k = (r["run"], r["name"])
        st[k][r["label"]] += 1
        par[k] = r
    labs = sorted({r["label"] for r in rows})
    cmap = plt.get_cmap("tab20")
    col = {lb: cmap(i % 20) for i, lb in enumerate(labs)}
    modes = [("shell 1", lambda p: p["Jex2"] == 0 and p["Jex3"] == 0 and p["Jex4"] == 0),
             ("Jex2 only", lambda p: p["Jex2"] != 0 and p["Jex3"] == 0 and p["Jex4"] == 0),
             ("Jex2-4", lambda p: p["Jex3"] != 0 or p["Jex4"] != 0)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for j, (title, f) in enumerate(modes):
        for i, (xk, yk) in enumerate((("KDM", "Hex"), ("Kan1", "KanS"))):
            a = axs[i, j]
            for lb in labs:
                pts = [(par[k][xk], par[k][yk]) for k in st if f(par[k]) and st[k].most_common(1)[0][0] == lb]
                if pts:
                    x, y = zip(*pts)
                    a.scatter(x, y, s=9, color=col[lb], label=lb)
            a.set_xlabel(xk); a.set_ylabel(yk); a.set_title(title); a.grid(alpha=.2)
    axs[0, 0].legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(out / "phase_map.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, imgs = [], []
    for ch in load(a.runs):
        N = ch["final"].shape[-1]
        mask = disk_mask(N)
        for i, nm in enumerate(ch["names"]):
            for r in range(ch["final"].shape[1]):
                d = descriptors(ch["final"][i, r], ch["Q"][i, r], mask)
                d.update(run=ch["run"], name=str(nm), replica=r, E=float(ch["E"][i, r]),
                         seconds=float(ch["seconds"][i]), colours=int(ch["colours"][i]),
                         **dict(zip(("Jex2", "Jex3", "Jex4", "Kan1", "KanS", "Hex", "KDM"),
                                    map(float, ch["params"][i]))))
                d["label"] = label(d)
                rows.append(d)
                imgs.append(np.where(mask, ch["final"][i, r, 2].astype(np.float32), np.nan))
    keys = list(rows[0].keys())
    with open(out / "descriptors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    labels = sorted({r["label"] for r in rows})
    summary = {}
    rng = np.random.default_rng(0)
    for lb in labels:
        idx = [i for i, r in enumerate(rows) if r["label"] == lb]
        states = sorted({(rows[i]["run"], rows[i]["name"]) for i in idx})
        P = np.array([[rows[i][k] for k in ("Jex2", "Jex3", "Jex4", "Kan1", "KanS", "Hex", "KDM")] for i in idx])
        summary[lb] = dict(replicas=len(idx), states=len(states),
                           p05=np.percentile(P, 5, 0).round(3).tolist(),
                           p50=np.percentile(P, 50, 0).round(3).tolist(),
                           p95=np.percentile(P, 95, 0).round(3).tolist())
        pick = sorted(rng.choice(idx, min(len(idx), a.per_class), replace=False),
                      key=lambda i: (rows[i]["KDM"], rows[i]["Hex"]))
        gallery([(imgs[i], f"{rows[i]['name'][-9:]} Q{rows[i]['Q']:+.1f}\nD{rows[i]['KDM']:.2f} H{rows[i]['Hex']:.2f}"
                  f" K{rows[i]['Kan1']:.2f}") for i in pick], out / f"gallery_{lb}.png",
                f"{lb}: {len(idx)} replicas, {len(states)} states")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    phase_map(rows, out)
    print(f"{len(rows)} replicas")
    for lb, s in summary.items():
        print(f"{lb:<20}{s['replicas']:>6} repl {s['states']:>5} states   median {s['p50']}")


if __name__ == "__main__":
    main()
