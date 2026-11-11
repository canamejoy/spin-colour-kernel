"""Parameter exploration of the nanodisk on Modal: which textures does each region of the seven
dataset parameters produce at low temperature?

Each state is cooled from T_START to T_END on the same linear ladder as the notebook, with the same
modules (build_state, Sampler, Observer). Only the final configuration matters here, so nothing is
measured during the ladder: at every temperature all sweeps thermalize. Per state it returns, for
every replica, the middle-layer spins (3 components) at the end and the middle-layer s_z at a few
snapshot temperatures, plus E/site, the magnetization vector and the Berg-Luscher charge Q.

    modal run remote/explore_nanodisk.py --params params.json --run-id pilot --chunks 4 \
        --replicas 16 --n-temps 60 --sweeps 1000

params.json holds {"names": [...], "params": [[Jex2, Jex3, Jex4, Kan1, KanS, Hex, KDM], ...]}.
States are grouped by their own active shells (one kernel each anyway), so a state with Jex2 = 0
keeps the 2-colour first-shell colouring. Output: results/explore/<run-id>/chunk_<i>.npz.
"""
import io
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("torch==2.5.1", "numpy")
         .env({"SCK_CACHE": "/tmp/sck_cache"})
         .add_local_dir(ROOT / "src", "/root/src"))
app = modal.App("sck-explore", image=image)

SNAP_T = (10.0, 5.0, 2.0)          # K: s_z snapshots on the way down, besides the final one


@app.function(gpu="H100", timeout=4 * 3600)
def run_chunk(rows: list, names: list, replicas: int, T_start: float, T_end: float, n_temps: int,
              sweeps: int, seed: int, Rd: float, layers: int, angle: float) -> bytes:
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np
    import torch
    import Spin_Colour_Kernel.anneal as AN
    from Spin_Colour_Kernel.sampler import Sampler

    dev = "cuda:0"
    temps = np.linspace(T_start, T_end, n_temps)
    snap_idx = [int(np.argmin(np.abs(temps - t))) for t in SNAP_T]
    out = {k: [] for k in ("final", "snaps", "E", "M", "Q", "seconds", "colours", "exact")}
    checked = set()
    for i, (row, nm) in enumerate(zip(rows, names)):
        shells = AN.active_shells([row])
        s = AN.build_state(row, Rd, layers, angle, shells)
        key = tuple(shells)
        if key not in checked:           # exactness depends on the colouring (the shells), not the values
            d = AN.check_exact(s, dev)
            if d:
                raise SystemExit(f"[{nm}] coloured kernel differs from the sequential chain: {d} spins")
            checked.add(key)
            out["exact"].append([list(key), d])
        t0 = time.perf_counter()
        A = Sampler(s, replicas, dev, seed + i)
        S = A.init_spins(seed=seed + i)
        A.set_temperature(AN.KB * float(temps[0]))
        blk = 100 if sweeps % 100 == 0 else sweeps
        rep = A.graph(S, blk)
        obs = AN.Observer(A)
        mid = obs.mid
        snaps = []
        for t, TK in enumerate(temps):
            A.set_temperature(AN.KB * float(TK))
            for _ in range(sweeps // blk):
                rep()
            if t in snap_idx:
                snaps.append(S[:, 2, 0, mid].half().cpu().numpy())
        E, Mv, Q = obs(S)
        torch.cuda.synchronize()
        out["final"].append(S[:, :, 0, mid].half().cpu().numpy())        # (R, 3, N, N)
        out["snaps"].append(np.stack(snaps, 1))                            # (R, n_snap, N, N)
        out["E"].append((E / s.n_active).cpu().numpy())
        out["M"].append(Mv.cpu().numpy())
        out["Q"].append(Q.cpu().numpy())
        out["seconds"].append(time.perf_counter() - t0)
        out["colours"].append(s.colouring.n_colours)
        print(f"[{nm}] shells {shells} {s.colouring.n_colours} col  {out['seconds'][-1]:6.1f} s  "
              f"Q {Q.mean().item():+6.2f}  Mz {Mv[:, 2].mean().item():+.3f}", flush=True)
        del A, S, rep, obs
    buf = io.BytesIO()
    np.savez_compressed(buf, params=np.asarray(rows, np.float32), names=np.asarray(names),
                        temps=temps, snap_T=np.asarray([temps[j] for j in snap_idx]),
                        **{k: np.asarray(v) for k, v in out.items() if k != "exact"},
                        exact=json.dumps(out["exact"]))
    return buf.getvalue()


@app.local_entrypoint()
def main(params: str, run_id: str, chunks: int = 1, replicas: int = 16, n_temps: int = 60,
         sweeps: int = 1000, t_start: float = 20.0, t_end: float = 0.1, seed: int = 7,
         rd: float = 18.30, layers: int = 5, angle: float = 90.0):
    cfg = json.loads(Path(params).read_text())
    rows, names = cfg["params"], cfg["names"]
    # balance the chunks by colour count (cost per sweep grows with it)
    cost = [len([v for v in r[:3] if v != 0]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -cost[i])
    parts = [[] for _ in range(chunks)]
    for j, i in enumerate(order):
        parts[j % chunks].append(i)
    dest = ROOT / "results" / "explore" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(dict(params_file=params, chunks=chunks, replicas=replicas,
                                                      n_temps=n_temps, sweeps=sweeps, t_start=t_start,
                                                      t_end=t_end, seed=seed, rd=rd, layers=layers,
                                                      angle=angle, parts=parts), indent=1))
    args = [([rows[i] for i in p], [names[i] for i in p], replicas, t_start, t_end, n_temps, sweeps,
             seed + 1000 * k, rd, layers, angle) for k, p in enumerate(parts) if p]
    t0 = time.time()
    for k, blob in enumerate(run_chunk.starmap(args, return_exceptions=True)):
        if isinstance(blob, Exception):
            print(f"chunk {k} failed: {blob}")
            continue
        (dest / f"chunk_{k}.npz").write_bytes(blob)
        print(f"chunk {k} saved ({len(blob) / 1e6:.1f} MB)")
    print(f"wall {time.time() - t0:.0f} s -> {dest}")
