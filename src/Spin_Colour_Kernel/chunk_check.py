"""E1chunk (experiments v17): on the GPU, rows longer than MAX_BLOCK run chunked over grid axis 2; the spins must
equal those of the one-program-per-row kernel (max_block large enough for the whole row) after several sweeps,
and the rate of both is reported (the unchunked kernel spills registers at large rows)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import crystals as cr
from . import system as sy
from .sampler import Sampler

CASES = (("heisenberg", "triangular", 2048), ("heisenberg", "triangular", 4096), ("xy", "square", 2048),
         ("heisenberg", "sc", 1152))


def build(kind, crystal, L):
    if crystal == "sc":
        return sy.System("Heisenberg sc long rows", 3, cr.make("sc"), (L, 8, 8), (True,) * 3, {1: 1.0})
    return (sy.heisenberg if kind == "heisenberg" else sy.xy)(crystal, L)


def rate(A, S, k=3):
    torch.cuda.synchronize()
    t = time.perf_counter()
    A.sweep(S, k)
    torch.cuda.synchronize()
    return A.sys.n_active * A.B * k / ((time.perf_counter() - t) * 1e9)


def chunk_check(out: Path, device="cuda:0", sweeps: int = 4, quick: bool = False):
    rows = []
    for kind, crystal, L in (CASES[:1] if quick else CASES):
        s = build(kind, crystal, 512 if quick else L)
        L1 = s.L1
        C = Sampler(s, 1, device, seed=7)
        U = Sampler(s, 1, device, seed=7, max_block=1 << (L1 - 1).bit_length())
        for X in (C, U):
            X.set_temperature(1.0)
        S1 = C.init_spins(seed=8)
        S2 = S1.clone()
        C.sweep(S1, sweeps)
        U.sweep(S2, sweeps)
        torch.cuda.synchronize()
        diff = int((S1 != S2).any(dim=1).sum().item())
        r = {"system": s.name, "box": list(s.box), "sites": s.n_active, "chunks": C.chunks, "cchunks": C.cchunks,
             "block": C.block, "unchunked_block": U.block, "sweeps": sweeps, "differing_sites": diff,
             "chunked_flips_per_ns": rate(C, S1), "unchunked_flips_per_ns": rate(U, S2)}
        rows.append(r)
        print(f"E1chunk {s.name} {tuple(s.box)} chunks={C.chunks} differing={diff} "
              f"rate chunked {r['chunked_flips_per_ns']:.3f}/ns unchunked {r['unchunked_flips_per_ns']:.3f}/ns", flush=True)
    res = {"experiment": "E1chunk", "rows": rows, "verdict": "supported" if all(r["differing_sites"] == 0 for r in rows)
           else "not_supported"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "E1chunk.json").write_text(json.dumps(res, indent=2))
    return res
