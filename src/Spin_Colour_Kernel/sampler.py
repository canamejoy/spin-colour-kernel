"""Launch plumbing for a generated kernel: sweeps, the sequential reference, observables.

A parallel sweep is n_c * b launches (colour-major, then sublattice). The sequential reference
visits the same classes in the same order and, inside each class, the sites one at a time in
raster order (z, y, x), through the same compiled kernel. Because no two sites of one class are
bonded (7), the two chains must hold identical spins after every sweep (hypothesis H1).
"""

from __future__ import annotations

import numpy as np
import torch

from . import bitwise as bw
from .generator import compile_system, x_period
from .system import System


#: smallest x-period worth compacting; E6 on a T4 (2026-11-07) lost 25-35% with stride-2 loads
COMPACT_MIN = 4
#: lanes a compacted program aims for: R rows of BX sites each
TILE = 128
#: float kernels: lanes per one-row program. A longer row is split over grid axis 2. Above 1024 lanes a 4-warp
#: Heisenberg program spills its registers to local memory (Nsight on the H100: 0, 18 M and 253 M local sectors at
#: rows of 1024, 2048 and 4096 sites), so rows up to 1024 sites keep one program, longer rows are chunked.
MAX_BLOCK = 1024
#: bit-packed Ising: lanes per program (each lane carries a 32-replica tile, so BLOCK x 32 values)
PTILE = 64
#: replicas per packed word
WORD = 32
#: bitwise: default words per program (one Philox call per site serves an aligned group of four words)
BWG = 4


def _num_warps(block: int) -> int:
    return 4 if block >= 256 else (2 if block >= 128 else 1)


class Sampler:
    def __init__(self, system: System, batch: int, device="cuda", seed: int = 0, compact: bool = True,
                 spin_dtype: torch.dtype | None = None, packed: bool | None = None, bitwise: bool = False,
                 words_per_program: int | None = None, independent: bool = False, bitwise_tile: int | None = None,
                 bitwise_warps: int | None = None, max_block: int = MAX_BLOCK):
        """bitwise: true multi-spin coding (module `bitwise`): packed storage updated with bit operations, one
        uniform per (word, site, sweep) shared by the word's 32 replicas, one temperature per word.
        independent (bitwise only): every replica draws its own 32-bit uniform U and flips iff U < floor(p 2^32),
        compared bit-sliced; replicas of a word no longer share randomness (p quantized down to 2^-32).
        words_per_program: bitwise words a program covers (power of two, default BWG; fewer when the batch is small)."""
        self.sys = system
        self.bitwise = bool(bitwise)
        self.independent = bool(independent)
        if self.independent and not self.bitwise:
            raise ValueError("independent per-replica draws need bitwise=True")
        if self.bitwise:
            _, _, self.zmax = bw.check(system)     # ValueError for systems the bitwise kernel cannot run
            if packed is False:
                raise ValueError("bitwise multi-spin coding uses the packed storage")
            packed = True
        # Ising defaults to multi-spin coding (32 replicas per int32 word, see generator.source) once the
        # batch fills a word; below that a word computes 32 replicas for fewer than 32 used, and int8 is faster
        self.packed = (system.n == 1 and int(batch) >= WORD) if packed is None else bool(packed)
        if self.packed and system.n != 1:
            raise ValueError("bit packing is for Ising (n = 1) only")
        # Ising spins are +-1 (0 on masked sites): int8 storage reads a quarter of the bytes of float32 and holds
        # them exactly; vector models keep float32 components
        self.spin_dtype = spin_dtype or (torch.int8 if system.n == 1 else torch.float32)
        self.B = int(batch)
        self.device = torch.device(device)
        self.kernel, self.src = compile_system(system, self.packed, self.bitwise, self.independent)
        n, b, L3, L2, L1 = system.n, system.b, system.L3, system.L2, system.L1
        self.W = -(-self.B // WORD) if self.packed else self.B      # program rows of grid axis 0
        self.Bp = self.W * WORD if self.packed else self.B           # replicas the kernel computes
        if self.Bp * b * L3 * L2 * L1 >= 2 ** 31 - 1:
            raise ValueError("replica batch too large for the int32 Philox counter")
        # bitwise: a program covers G words, a power of two; G >= 4 wastes no Philox output
        gmax = words_per_program or BWG
        if gmax & (gmax - 1):
            raise ValueError("words_per_program must be a power of two")
        self.G = min(gmax, 1 << (self.W - 1).bit_length()) if self.bitwise else 1
        self.ngroups = -(-self.W // self.G)
        self.nrows = L3 * L2
        if max_block < 16 or max_block & (max_block - 1):
            raise ValueError("max_block must be a power of two >= 16")
        self.block = min(max(16, 1 << (L1 - 1).bit_length()), max_block)
        self.chunks = -(-L1 // self.block)            # one-row programs over grid axis 2 (rows longer than max_block)
        self.nw = _num_warps(self.block)
        # colour compaction: a program covers R rows x BX sites of one colour (BX = ceil(L1 / P))
        P = x_period(system.colouring) if compact else 1
        P = P if P >= COMPACT_MIN else 1       # below it, strided loads cost more than the masked lanes save
        self.stride = P
        self.bx = -(-L1 // P)                         # sites of one colour per row
        tile_c = bitwise_tile if (bitwise and bitwise_tile) else TILE   # bitwise: lanes per compacted program
        self.rows_per = max(1, min(self.nrows, tile_c // self.bx)) if P > 1 else 1
        if self.packed and not self.bitwise:   # a bitwise lane holds G int32 words, not a 32-float tile
            self.rows_per = max(1, min(self.nrows, PTILE // self.bx)) if P > 1 else 1
        self.cblock = max(16, 1 << (self.rows_per * self.bx - 1).bit_length())
        self.cchunks = 1
        if not self.packed and self.rows_per == 1 and self.cblock > max_block:
            self.cblock = max_block
            self.cchunks = -(-self.bx // max_block)
        self.cnw = _num_warps(self.cblock)
        if self.packed:   # one-row programs cover at most PTILE sites of the row; longer rows go over grid axis 2
            nx = self.bx if P > 1 else L1
            tile = TILE if self.bitwise else PTILE
            self.pblock = min(tile, max(16, 1 << (nx - 1).bit_length()))
            self.pchunks = -(-nx // self.pblock)
            self.pblock_ref = min(tile, self.block)
            self.pchunks_ref = -(-L1 // self.pblock_ref)
            if not self.bitwise:
                self.cnw = 4 if self.cblock * WORD >= 2048 else 2
            elif bitwise_tile:
                self.pblock = min(bitwise_tile, max(16, 1 << (nx - 1).bit_length()))
                self.pchunks = -(-nx // self.pblock)
            if self.bitwise and bitwise_warps:     # tuning knob of the bitwise kernels (E6 pilot)
                self.cnw = self.nw = int(bitwise_warps)
        dev = self.device
        m = system.site_mask()
        self.mask_np = m
        self.mask = torch.as_tensor(m.astype(np.int8), device=dev)
        k1 = system.cubic_k1 if system.cubic_k1 is not None else np.zeros(m.shape, np.float32)
        self.k1 = torch.as_tensor(np.asarray(k1, np.float32), device=dev)
        self.state = torch.tensor([int(seed), 0], dtype=torch.int64, device=dev)
        self.beta = torch.ones(self.Bp, dtype=torch.float32, device=dev)
        if self.bitwise:     # acceptance table p[word, s, z_eff, k], rebuilt in place by set_temperature
            self._table = bw.p_fixed if self.independent else bw.p_table     # int64 P = floor(p 2^32), or float32 p
            self.ptab = torch.as_tensor(self._table(system, np.ones(self.W)), device=dev)
        self.colours = system.colour_map()
        self.classes = [(c, a) for c in range(system.colouring.n_colours) for a in range(b)
                        if np.any((self.colours[a] == c) & m[a])]
        mt = torch.as_tensor(m, device=dev)
        self._mask_f = mt.to(torch.float32)

    # ------------------------------------------------------------ state
    def set_temperature(self, T):
        T = torch.as_tensor(T, dtype=torch.float32, device=self.device).reshape(-1)
        beta = (1.0 / T).expand(self.B).contiguous() if T.numel() == 1 else (1.0 / T).contiguous()
        if self.Bp > self.B:     # padding replicas of the last word: computed, never observed
            beta = torch.cat([beta, beta[-1:].expand(self.Bp - self.B)])
        if self.bitwise:
            bwd = beta.reshape(self.W, WORD)
            if not torch.all(bwd == bwd[:, :1]):
                raise ValueError("bitwise multi-spin coding needs one temperature per word of 32 replicas")
            # in place, like beta: a captured CUDA graph reads the table by address
            self.ptab.copy_(torch.as_tensor(self._table(self.sys, bwd[:, 0].double().cpu().numpy()), device=self.device))
        # in place: a CUDA graph captured earlier reads this buffer by address, so a new tensor would be ignored
        self.beta.copy_(beta)

    def set_seed(self, seed: int, sweep: int = 0):
        self.state[0] = int(seed)
        self.state[1] = int(sweep)

    def init_spins(self, kind="random", seed: int = 0) -> torch.Tensor:
        s = self.sys
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        shape = (self.B, s.n) + s.shape
        if kind == "ordered":
            x = torch.zeros(shape)
            x[:, 0] = 1.0
        elif s.n == 1:
            x = torch.randint(0, 2, shape, generator=g).float() * 2 - 1
        else:
            x = torch.randn(shape, generator=g)
            x = x / x.norm(dim=1, keepdim=True)
        x = x.to(self.device) * self._mask_f
        if self.packed:
            return self.pack(x)
        return x.to(self.spin_dtype).contiguous()

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, b, L3, L2, L1) spins of +-1 (0 on masked sites) -> (W, 1, b, L3, L2, L1) int32 words."""
        bits = (x > 0).to(torch.int64)
        if self.Bp > self.B:
            bits = torch.cat([bits, bits.new_zeros((self.Bp - self.B,) + tuple(bits.shape[1:]))])
        bits = bits.reshape((self.W, WORD) + tuple(bits.shape[1:]))
        sh = (torch.arange(WORD, device=bits.device, dtype=torch.int64)).reshape((1, WORD) + (1,) * (bits.dim() - 2))
        word = (bits << sh).sum(dim=1)
        word = torch.where(word >= 2 ** 31, word - 2 ** 32, word)
        return word.to(torch.int32).contiguous()

    def unpack(self, S: torch.Tensor) -> torch.Tensor:
        """Packed words -> (B, 1, b, L3, L2, L1) int8 spins of +-1, 0 on masked sites; identity when unpacked."""
        if not self.packed or S.dtype != torch.int32:     # already spins
            return S
        w = S.to(torch.int64) & 0xFFFFFFFF
        sh = torch.arange(WORD, device=S.device, dtype=torch.int64).reshape((1, WORD) + (1,) * (S.dim() - 1))
        bits = (w.unsqueeze(1) >> sh) & 1
        x = (bits * 2 - 1).reshape((self.Bp,) + tuple(S.shape[1:]))[: self.B]
        return (x * self._mask_f.to(torch.int64)).to(torch.int8)

    # ------------------------------------------------------------ sweeps
    def _launch(self, S, dst, c, a, grid_rows, row0=0, x_only=-1, sync=False, compact=False):
        if self.bitwise:
            if compact and self.rows_per > 1:
                grid = (self.ngroups, -(-grid_rows // self.rows_per), 1)
                kw = dict(BLOCK=self.cblock, STRIDE=self.stride, BX=self.bx, R=self.rows_per, num_warps=self.cnw)
            elif compact:
                grid = (self.ngroups, grid_rows, self.pchunks)
                kw = dict(BLOCK=self.pblock, STRIDE=self.stride, BX=self.pblock, R=1, num_warps=self.nw)
            else:
                grid = (self.ngroups, grid_rows, self.pchunks_ref)
                kw = dict(BLOCK=self.pblock_ref, STRIDE=1, BX=self.pblock_ref, R=1, num_warps=self.nw)
            args = (S, dst, self.mask, self.k1, self.ptab, self.state, x_only, row0, self.W)
            kw["G"] = self.G
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self.kernel[grid](*args, ALPHA=a, COLOUR=c, SYNC=sync, **kw)
                return
            self.kernel[grid](*args, ALPHA=a, COLOUR=c, SYNC=sync, **kw)
            return
        if self.packed:
            if compact and self.rows_per > 1:
                grid = (self.W, -(-grid_rows // self.rows_per), 1)
                kw = dict(BLOCK=self.cblock, STRIDE=self.stride, BX=self.bx, R=self.rows_per, num_warps=self.cnw)
            elif compact:
                grid = (self.W, grid_rows, self.pchunks)
                kw = dict(BLOCK=self.pblock, STRIDE=self.stride, BX=self.pblock, R=1, num_warps=4)
            else:
                grid = (self.W, grid_rows, self.pchunks_ref)
                kw = dict(BLOCK=self.pblock_ref, STRIDE=1, BX=self.pblock_ref, R=1, num_warps=4)
        elif compact:
            grid = (self.B, -(-grid_rows // self.rows_per), self.cchunks)
            kw = dict(BLOCK=self.cblock, STRIDE=self.stride, BX=self.bx, R=self.rows_per, num_warps=self.cnw)
        else:
            grid = (self.B, grid_rows, self.chunks)
            kw = dict(BLOCK=self.block, STRIDE=1, BX=self.block, R=1, num_warps=self.nw)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.kernel[grid](S, dst, self.mask, self.k1, self.beta, self.state, x_only, row0,
                                  ALPHA=a, COLOUR=c, SYNC=sync, **kw)
            return
        self.kernel[grid](S, dst, self.mask, self.k1, self.beta, self.state, x_only, row0,
                          ALPHA=a, COLOUR=c, SYNC=sync, **kw)

    def sweep(self, S: torch.Tensor, n: int = 1) -> torch.Tensor:
        """n coloured sweeps in place; advances the sweep counter."""
        for _ in range(n):
            for c, a in self.classes:
                self._launch(S, S, c, a, self.nrows, compact=self.stride > 1)
            self.state[1] += 1
        return S

    def graph(self, S: torch.Tensor, k: int):
        """Capture k sweeps (launches and counter increments) in a CUDA graph; returns its replay.

        Removes the host launch cost from timings; the counter stream is the same as k calls of
        `sweep`, because the sweep counter is incremented on the device inside the graph.
        """
        with torch.cuda.device(self.device):
            return self._graph(S, k)

    def _graph(self, S: torch.Tensor, k: int):
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            self.sweep(S, 1)          # warm-up compiles every class launch outside the capture
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.sweep(S, k)
        dev = self.device

        def replay():
            with torch.cuda.device(dev):
                g.replay()
        # the graph addresses this sampler's buffers and S directly: the replay keeps them alive, or a caller
        # that drops the sampler (or rebinds S) before replaying reads freed memory
        replay.keep = (self, S, g)
        return replay

    def sync_sweep(self, S: torch.Tensor, n: int = 1) -> torch.Tensor:
        """The negative control of E2: every site updated in the same step from the old state."""
        for _ in range(n):
            dst = S.clone()
            for a in range(self.sys.b):
                self._launch(S, dst, -1, a, self.nrows, sync=True)
            S.copy_(dst)
            self.state[1] += 1
        return S

    def reference_sweep(self, S: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Sequential reference chain: one launch per site, same kernel, same counters."""
        L2, L1 = self.sys.L2, self.sys.L1
        order = []
        for c, a in self.classes:
            sites = np.argwhere((self.colours[a] == c) & self.mask_np[a])   # (z, y, x), raster order
            order.append((c, a, sites))
        for _ in range(n):
            for c, a, sites in order:
                for z, y, x in sites:
                    self._launch(S, S, c, a, 1, row0=int(z) * L2 + int(y), x_only=int(x))
            self.state[1] += 1
        return S

    # ------------------------------------------------------------ observables
    def local_field(self, S: torch.Tensor) -> torch.Tensor:
        """Bond part of the local field (3) for every site, by rolls (float64), as a reference."""
        s = self.sys
        D = s.crystal.D
        S = self.unpack(S)
        S64 = S.double()
        v = torch.zeros_like(S64)
        dims = {0: 5, 1: 4, 2: 3}       # lattice axis k -> tensor dim
        for i, bd in enumerate(s.crystal.bonds):
            M = torch.as_tensor(s.bond_matrix(i), dtype=torch.float64, device=S.device)
            if not M.any():
                continue
            src = S64[:, :, bd.alpha2]
            valid = torch.ones(s.shape[1:], dtype=torch.float64, device=S.device)
            for k in range(D):
                d = bd.delta[k]
                if d == 0:
                    continue
                src = torch.roll(src, shifts=-d, dims=dims[k] - 1)
                if not s.periodic[k]:
                    Lk = s.box[k]
                    idx = torch.arange(Lk, device=S.device)
                    ok = ((idx + d) >= 0) & ((idx + d) < Lk)
                    shp = [1, 1, 1]
                    shp[2 - k] = Lk
                    valid = valid * ok.to(torch.float64).reshape(shp)
            src = src * valid
            v[:, :, bd.alpha] += torch.einsum("kl,blzyx->bkzyx", M, src)
        return v

    def energy(self, S: torch.Tensor) -> torch.Tensor:
        """Total energy per replica, eq. (1), float64."""
        s = self.sys
        S = self.unpack(S)
        S64 = S.double()
        v = self.local_field(S)
        E = -0.5 * (S64 * v).sum(dim=(1, 2, 3, 4, 5))
        if s.h is not None:
            h = torch.as_tensor(s.h, dtype=torch.float64, device=S.device).reshape(1, -1, 1, 1, 1, 1)
            E = E - (S64 * h).sum(dim=(1, 2, 3, 4, 5))
        if s.cubic_k1 is not None:
            x, y, z = S64[:, 0], S64[:, 1], S64[:, 2]
            k1 = self.k1.double()
            a = k1 * ((x * y) ** 2 + (x * z) ** 2 + (y * z) ** 2) + s.cubic_k2 * (x * y * z) ** 2
            E = E + a.sum(dim=(1, 2, 3, 4))
        return E

    def magnetization(self, S: torch.Tensor) -> torch.Tensor:
        """Per-sublattice magnetization M_alpha, shape (B, b, n), float64."""
        return self.unpack(S).double().sum(dim=(3, 4, 5)).transpose(1, 2)
