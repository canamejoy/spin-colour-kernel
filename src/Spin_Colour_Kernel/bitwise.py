"""Bitwise multi-spin-coded Ising: asynchronous multi-spin coding over replicas (routes B and C).

Storage is the packed word layout of `generator.source(packed=True)`: word w of a site holds that site's
spin in replicas 32 w .. 32 w + 31, one bit each (1 = +1). Unlike the packed path, which unpacks every word
to a [BLOCK, 32] float tile, this kernel never leaves the bits:

* per neighbour j present in the box and the mask, x = s_i XOR s_j marks the anti-aligned replicas, and a
  ripple adder accumulates x into NP bit planes, so plane p holds bit p of the anti-aligned count k of every
  replica (NP = bit length of the largest coordination, 4 for fcc);
* the energy change of flipping s_i is dE = 2 J (z_eff - 2 k) + 2 h s_i, with z_eff the number of present
  neighbours of the site (the same for all 32 replicas), so the acceptance probability is a host table
  p[word, s, z_eff, k] built per temperature for the system's rule (Metropolis min(1, e^{-beta dE}) or
  Barker 1 / (1 + e^{beta dE}));
* one uniform u per (word, site, sweep) is shared by the 32 replicas of the word, and the flip mask is
  OR_k [(u < p(z_eff, k[, s])) ? eq_k : 0], with eq_k the replicas whose count equals k.

Each replica is still an exact Metropolis or Barker chain; replicas of one word are correlated through
the shared uniform (and may coalesce), so independent samples come from different words.

Randomness: Philox keyed by the per-sweep seed of the other kernels. Uniform (word w, site i) is output
w % 4 of the call with counter (w // 4) * nsite + i, so one call serves four words and a program that
covers an aligned group of four words wastes no output. The stream is a function of (word, site, sweep)
only, never of the launch grid; `uniforms` reproduces it in NumPy for the sequential reference.

Independent mode (`source(sys_, independent=True)`, `Sampler(..., bitwise=True, independent=True)`): the
shared uniform makes the replicas of a word coalesce (at Tc on the square lattice 32 replicas collapse to two
configurations within about 50 sweeps), so this mode gives every replica (bit lane) its own draw and keeps the
update exact and bitwise:

* the table holds P = floor(p 2^32) as an integer in [0, 2^32] (`p_fixed`); a replica flips iff its 32-bit
  uniform integer U satisfies U < P, an exact Bernoulli(P / 2^32) draw. p is therefore quantized down to a
  multiple of 2^-32; p = 1 gives P = 2^32 (always, no random bits) and P = 0 never flips;
* U is compared bit-sliced over the 32 lanes, most significant bit first: at level l a random word R_l holds
  bit 31 - l of U of every lane and Pb_l the same bit of every lane's P (selected by the count planes, and by
  s when h != 0); accept |= und & ~R_l & Pb_l, und &= ~(R_l ^ Pb_l). Lanes start undecided iff 0 < P < 2^32;
  each level halves them, so the program exits as soon as its whole tile is decided, and after 32 levels the
  rest (U == P) is rejected, exactly;
* R_{4g} .. R_{4g+3} of (word w, site i) are the four raw outputs of `tl.randint4x(key_g, w nsite + i)` with
  the int32 counter of the shared mode and the per-group key key_g = (seed + (g + 1) GROUP_KEY_STEP) mod
  2147483629, a function of (word, site, level group, sweep) only; `lane_uniforms` rebuilds every replica's U
  in NumPy. (A 64-bit offset would switch Triton's GPU randint4x to 64-bit Philox, which the interpreter does
  not do, so the level group goes into the key, not into the counter.)

Only systems with one Ising coupling J on every bond that carries one (any crystal, any shells of equal J),
an optional uniform field and an optional site mask qualify; `check` refuses the others.
"""

from __future__ import annotations

import numpy as np

from .system import System

#: replicas per word
WORD = 32


# ------------------------------------------------------------------ system
def check(sys_: System) -> tuple[float, float, int]:
    """(J, h, Z) of a system the bitwise kernel can run, Z the largest coordination; ValueError otherwise."""
    if sys_.n != 1:
        raise ValueError("bitwise multi-spin coding is for Ising (n = 1) only")
    if sys_.extra:
        raise ValueError("bitwise multi-spin coding needs isotropic exchange only (no extra bond matrices)")
    if sys_.cubic_k1 is not None:
        raise ValueError("bitwise multi-spin coding does not support anisotropy")
    Js = {float(sys_.J.get(bd.shell, 0.0)) for bd in sys_.crystal.bonds} - {0.0}
    if len(Js) != 1:
        raise ValueError(f"bitwise multi-spin coding needs one uniform coupling J on every bond, got {sorted(Js)}")
    h = float(sys_.h[0]) if sys_.h is not None else 0.0
    return Js.pop(), h, max(len(bonds(sys_, a)) for a in range(sys_.b))


def bonds(sys_: System, a: int) -> list:
    """Bonds leaving sublattice a that carry a coupling, in bond-table order."""
    return [bd for bd in sys_.crystal.bonds if bd.alpha == a and sys_.J.get(bd.shell, 0.0) != 0.0]


def p_table(sys_: System, beta: np.ndarray) -> np.ndarray:
    """Acceptance probabilities p[word, s, z_eff, k], float32, for one inverse temperature per word.

    s = 0 for spin -1 and 1 for +1, z_eff the number of present neighbours, k <= z_eff the anti-aligned
    ones; entries with k > z_eff are never read and hold 0."""
    return _p64(sys_, beta).astype(np.float32)


def p_fixed(sys_: System, beta: np.ndarray) -> np.ndarray:
    """Independent mode: P = floor(p 2^32) in [0, 2^32], int64, same layout as `p_table` (p in float64).

    A replica with 32-bit uniform integer U flips iff U < P: probability P / 2^32, i.e. p rounded down to a
    multiple of 2^-32 (P = 2^32 when p = 1: always; P = 0: never)."""
    P = np.floor(_p64(sys_, beta) * 2.0 ** 32).astype(np.int64)
    return np.clip(P, 0, 2 ** 32)


def _p64(sys_: System, beta: np.ndarray) -> np.ndarray:
    J, h, Z = check(sys_)
    beta = np.asarray(beta, np.float64).reshape(-1)
    p = np.zeros((beta.size, 2, Z + 1, Z + 1), np.float64)
    for s in (0, 1):
        sigma = 2 * s - 1
        for ze in range(Z + 1):
            k = np.arange(ze + 1)
            dE = 2.0 * J * (ze - 2 * k) + 2.0 * h * sigma
            x = beta[:, None] * dE[None, :]
            if sys_.acceptance == "barker":
                p[:, s, ze, : ze + 1] = 0.5 * (1.0 - np.tanh(0.5 * x))
            else:
                p[:, s, ze, : ze + 1] = np.exp(-np.maximum(x, 0.0))
    return p


# ------------------------------------------------------------------ randomness
_M = 0xFFFFFFFF


def philox_uniform4(seed: int, counters) -> np.ndarray:
    """NumPy twin of Triton's `tl.rand4x(seed, counters)` for an int32 seed and int32 counters: (4, N) float32."""
    out = philox_raw4(seed, counters).view(np.int32)
    out = np.where(out < 0, ~out, out)
    return out.astype(np.float32) * np.float32(4.6566127342e-10)


def philox_raw4(seed: int, c0, c1=0, rounds: int = 10) -> np.ndarray:
    """NumPy twin of Triton's `tl.randint4x(seed, off)` with off = c1 2^32 + c0 (c0 < 2^32): (4, N) uint32.

    Philox-4x32-10 with key (seed low, seed high) and counter (c0, c1, 0, 0)."""
    c0 = np.asarray(c0, np.int64).reshape(-1)
    c1 = np.broadcast_to(np.asarray(c1, np.int64).reshape(-1), c0.shape)
    c0, c1 = (c0 & _M).astype(np.uint64), (c1 & _M).astype(np.uint64)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0 = np.uint64(int(seed) & _M)
    k1 = np.uint64((int(seed) >> 32) & _M)
    A, B = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    m, s32 = np.uint64(_M), np.uint64(32)
    for _ in range(rounds):
        p0, p2 = B * c2, A * c0           # < 2^64: exact in uint64
        c0, c1, c2, c3 = ((p0 >> s32) ^ c1 ^ k0) & m, p0 & m, ((p2 >> s32) ^ c3 ^ k1) & m, p2 & m
        k0 = (k0 + np.uint64(0x9E3779B9)) & m
        k1 = (k1 + np.uint64(0xBB67AE85)) & m
    return np.stack([c0, c1, c2, c3]).astype(np.uint32)


def sweep_seed(base_seed: int, sweep: int) -> int:
    """The per-sweep Philox key of every generated kernel."""
    return ((int(base_seed) * 1000003 + int(sweep) * 7919 + 12345) % 2147483629)


def uniforms(base_seed: int, sweep: int, n_words: int, nsite: int) -> np.ndarray:
    """The kernel's uniform of every (word, site) in one sweep: shape (n_words, nsite), float32."""
    groups = -(-n_words // 4)
    ctr = np.arange(groups, dtype=np.int64)[:, None] * nsite + np.arange(nsite, dtype=np.int64)[None, :]
    u = philox_uniform4(sweep_seed(base_seed, sweep), ctr.reshape(-1)).reshape(4, groups, nsite)
    return u.transpose(1, 0, 2).reshape(groups * 4, nsite)[:n_words]


#: independent mode: random words per (word, site, sweep), one per bit of U, four per Philox call
LEVELS = 32
#: independent mode: the Philox key of level group g is (seed + (g + 1) GROUP_KEY_STEP) mod 2147483629
GROUP_KEY_STEP = 104729
#: independent mode: Philox-4x32 rounds per call, Triton's default 10 (the kernel and the NumPy twin agree); 7 rounds
#: is about 15% faster and is kept as an ablation (experiments v11/v12), not as the default
INDEPENDENT_ROUNDS = 10


def group_key(seed: int, g: int) -> int:
    return (int(seed) + (int(g) + 1) * GROUP_KEY_STEP) % 2147483629


def lane_uniforms(base_seed: int, sweep: int, n_words: int, nsite: int) -> np.ndarray:
    """Independent mode: the 32-bit uniform integer U of every (replica, site) in one sweep, (32 n_words, nsite)
    int64. Bit 31 - l of U of replica 32 w + j is bit j of the random word R_l(w, site), R_{4g + q} being
    output q of the Philox call with key `group_key(seed, g)` and counter w nsite + site."""
    ctr = (np.arange(n_words, dtype=np.int64)[:, None] * nsite + np.arange(nsite, dtype=np.int64)[None, :]).reshape(-1)
    seed = sweep_seed(base_seed, sweep)
    lanes = np.arange(WORD, dtype=np.uint64)
    U = np.zeros((n_words, WORD, nsite), np.int64)
    for g in range(LEVELS // 4):
        R = philox_raw4(group_key(seed, g), ctr, rounds=INDEPENDENT_ROUNDS).astype(np.uint64).reshape(4, n_words, 1, nsite)
        for q in range(4):
            bit = ((R[q] >> lanes[None, :, None]) & np.uint64(1)).astype(np.int64)     # (n_words, 32, nsite)
            U |= bit << (LEVELS - 1 - (4 * g + q))
    return U.reshape(n_words * WORD, nsite)


# ------------------------------------------------------------------ sequential reference
def classes(sys_: System) -> list[tuple[int, int]]:
    """(colour, sublattice) classes in sweep order: colour-major, then sublattice; empty classes skipped."""
    cm, m = sys_.colour_map(), sys_.site_mask()
    return [(c, a) for c in range(sys_.colouring.n_colours) for a in range(sys_.b) if np.any((cm[a] == c) & m[a])]


def reference_sweep(sys_: System, spins: np.ndarray, ptab: np.ndarray, base_seed: int, sweep: int,
                    independent: bool = False) -> np.ndarray:
    """One sequential sweep of every replica on unpacked spins, no bit operations.

    spins: (B, b, L3, L2, L1) of +-1 (0 on masked sites). For each class in sweep order and each of its sites
    in raster order, every replica r computes its local sum sigma_i sum_j s_j over the present neighbours,
    k = (z_eff - that sum) / 2, draws the uniform of (word r // 32, site, sweep) and flips when
    u < ptab[r // 32, s, z_eff, k]. independent: ptab is the integer table of `p_fixed` and replica r flips iff
    its own integer U (`lane_uniforms`) satisfies U < ptab[r // 32, s, z_eff, k]. Returns the updated copy."""
    _, h, _ = check(sys_)
    S = np.array(spins, dtype=np.int64, copy=True)
    B = S.shape[0]
    b, L3, L2, L1 = sys_.shape
    D = sys_.crystal.D
    box = (L1, L2, L3)
    per = list(sys_.periodic) + [False] * (3 - D)
    m = sys_.site_mask()
    cm = sys_.colour_map()
    nsite = b * L3 * L2 * L1
    W = -(-B // WORD)
    if independent:
        UL = lane_uniforms(base_seed, sweep, W, nsite)
    else:
        U = uniforms(base_seed, sweep, W, nsite)
    word = np.arange(B) // WORD
    plane = 1 if h != 0.0 else None
    for c, a in classes(sys_):
        for z, y, x in np.argwhere((cm[a] == c) & m[a]):
            nb = []
            for bd in bonds(sys_, a):
                d = list(bd.delta) + [0] * (3 - D)
                q = [x + d[0], y + d[1], z + d[2]]
                ok = True
                for k in range(3):
                    if per[k]:
                        q[k] %= box[k]
                    elif not 0 <= q[k] < box[k]:
                        ok = False
                if ok and m[bd.alpha2, q[2], q[1], q[0]]:
                    nb.append((bd.alpha2, q[2], q[1], q[0]))
            ze = len(nb)
            si = S[:, a, z, y, x]
            tot = sum((S[:, a2, z2, y2, x2] for a2, z2, y2, x2 in nb), np.zeros(B, np.int64))
            kk = (ze - si * tot) // 2
            s_idx = (si > 0).astype(np.int64) if plane else np.zeros(B, np.int64)
            p = ptab[word, s_idx, ze, kk]
            site = a * L3 * L2 * L1 + z * L2 * L1 + y * L1 + x
            if independent:
                u = UL[:B, site]                           # one integer per replica, compared as an integer
            else:
                u = U[word, site]
            S[:, a, z, y, x] = np.where(u < p, -si, si)
    return S.astype(spins.dtype)


# ------------------------------------------------------------------ kernel source
def source(sys_: System, independent: bool = False) -> str:
    """Triton source of the bitwise kernel; same launch structure and class predicate as `generator.source`.

    Grid axis 0 enumerates blocks of G words (G a power of two, a constexpr). One Philox call per site
    lane and aligned group of four words yields the uniforms of those four words; for G > 4 the G // 4
    calls of a lane are interleaved back to word order with tl.join.

    independent: one 32-bit uniform per replica, compared bit-sliced with the integer table of `p_fixed`
    (module docstring); p_ptr then points at that int64 table."""
    from .generator import _coord_expr, _emit_class, _emit_tile

    J, h, Z = check(sys_)
    b, L1, L2, L3 = sys_.b, sys_.L1, sys_.L2, sys_.L3
    D = sys_.crystal.D
    per = list(sys_.periodic) + [False] * (3 - D)
    pl = L3 * L2 * L1
    nsite = b * pl
    has_mask = sys_.mask is not None
    NP = max(1, Z.bit_length())
    Z1 = Z + 1
    NT = 2 * Z1 * Z1
    lines: list[str] = []
    w = lines.append
    w('"""Bitwise multi-spin-coded Ising kernel (generated by Spin_Colour_Kernel.bitwise)."""')
    w("import triton")
    w("import triton.language as tl")
    w("")
    w("")
    w('@triton.jit(do_not_specialize=["x_only", "row0", "nword"])')
    w("def kernel(src_ptr, dst_ptr, mask_ptr, k1_ptr, p_ptr, state_ptr, x_only, row0, nword,")
    w("           ALPHA: tl.constexpr, COLOUR: tl.constexpr, BLOCK: tl.constexpr, SYNC: tl.constexpr,")
    w("           STRIDE: tl.constexpr, BX: tl.constexpr, R: tl.constexpr, G: tl.constexpr):")
    w("    grp = tl.program_id(0)")
    w("    cw = tl.arange(0, G)")
    w("    wv = grp * G + cw")
    w("    wok = wv < nword")
    _emit_tile(w, sys_, True)
    w("    base_seed = tl.load(state_ptr + 0)")
    w("    sweep = tl.load(state_ptr + 1)")
    w("    seed = ((base_seed * 1000003 + sweep * 7919 + 12345) % 2147483629).to(tl.int32)")
    w(f"    rbase = wv.to(tl.int64) * {nsite}")
    _emit_class(w, sys_)
    w("    ldw = ld[:, None] & wok[None, :]")
    w("    s0 = tl.load(src_ptr + rbase[None, :] + own[:, None], mask=ldw, other=0)")
    w("    ze = tl.zeros([BLOCK], dtype=tl.int32)")
    for p in range(NP):
        w(f"    c{p} = tl.zeros([BLOCK, G], dtype=tl.int32)")
    for a in range(b):
        kw = "if" if a == 0 else "elif"
        w(f"    {kw} ALPHA == {a}:")
        bl = bonds(sys_, a)
        if not bl:
            w("        pass")
        for bd in bl:
            d = list(bd.delta) + [0] * (3 - D)
            ex, vx = _coord_expr("xc", d[0], L1, per[0])
            ey, vy = _coord_expr("yv", d[1], L2, per[1])
            ez, vz = _coord_expr("zv", d[2], L3, per[2])
            valid = " & ".join(["ld"] + [p for p in (vx, vy, vz) if p])
            w(f"        q = {bd.alpha2 * pl} + {ez} * {L2 * L1} + {ey} * {L1} + {ex}")
            w(f"        ok = {valid}")
            w("        q = tl.where(ok, q, 0)")
            if has_mask:      # a masked neighbour holds no spin: it is not a neighbour at all
                w("        ok = ok & (tl.load(mask_ptr + q, mask=ok, other=0) != 0)")
            w("        ze = ze + ok.to(tl.int32)")
            w("        tw = tl.load(src_ptr + rbase[None, :] + q[:, None], mask=ok[:, None] & wok[None, :], other=0)")
            w("        cy = tl.where(ok[:, None], s0 ^ tw, 0)")
            # ripple adder: plane p receives the carry out of plane p - 1
            for p in range(NP):
                if p < NP - 1:
                    w(f"        t = c{p} & cy")
                    w(f"        c{p} = c{p} ^ cy")
                    w("        cy = t")
                else:
                    w(f"        c{p} = c{p} ^ cy")
    if independent:
        _emit_independent(w, sys_, nsite, NP, Z1, NT, h != 0.0)
    else:
        _emit_shared(w, nsite, NP, Z1, NT, h != 0.0)
    w("    r0 = s0 ^ acc")
    w("    if SYNC:")
    wm = "live & (m_own != 0)" if has_mask else "live"
    w(f"        tl.store(dst_ptr + rbase[None, :] + own[:, None], r0, mask=({wm})[:, None] & wok[None, :])")
    w("    else:")
    w("        tl.store(dst_ptr + rbase[None, :] + own[:, None], r0, mask=upd[:, None] & wok[None, :])")
    return "\n".join(lines) + "\n"


def _emit_shared(w, nsite: int, NP: int, Z1: int, NT: int, field: bool) -> None:
    """Flip mask acc from one uniform per (word, site) shared by the word's 32 replicas."""
    # one Philox call per site lane: output w % 4 of counter (w // 4) * nsite + site
    w("    if G <= 4:")      # the program's words lie in one group of four
    w(f"        off = (((grp * G) // 4) * {nsite} + own).to(tl.int32)")
    w("        u0, u1, u2, u3 = tl.rand4x(seed, off)")
    w("        cq = (wv % 4)[None, :]")
    w("        u = tl.where(cq == 0, u0[:, None], tl.where(cq == 1, u1[:, None], tl.where(cq == 2, u2[:, None], u3[:, None])))")
    w("    else:")           # G // 4 whole groups: one call per (site, group), outputs interleaved to word order
    w(f"        gq = grp * (G // 4) + tl.arange(0, G // 4)")
    w(f"        off = (gq[None, :] * {nsite} + own[:, None]).to(tl.int32)")
    w("        u0, u1, u2, u3 = tl.rand4x(seed, off)")
    w("        u = tl.reshape(tl.join(tl.join(u0, u2), tl.join(u1, u3)), [BLOCK, G])")
    w(f"    tb = p_ptr + wv[None, :] * {NT} + ze[:, None] * {Z1}")
    w("    acc = tl.zeros([BLOCK, G], dtype=tl.int32)")
    for k in range(Z1):
        terms = [f"c{p}" if (k >> p) & 1 else f"~c{p}" for p in range(NP)]
        w(f"    eq = {' & '.join(terms)}")
        if field:
            w(f"    pm = tl.load(tb + {k}, mask=ldw, other=0.0)")
            w(f"    pp = tl.load(tb + {Z1 * Z1 + k}, mask=ldw, other=0.0)")
            w("    acc = acc | tl.where(u < pp, eq & s0, 0) | tl.where(u < pm, eq & ~s0, 0)")
        else:
            w(f"    pk = tl.load(tb + {k}, mask=ldw, other=0.0)")
            w("    acc = acc | tl.where(u < pk, eq, 0)")


def _emit_independent(w, sys_: System, nsite: int, NP: int, Z1: int, NT: int, field: bool) -> None:
    """Flip mask acc from one 32-bit uniform per replica, compared bit-sliced with P = floor(p 2^32).

    Table entries (one per k, and per s when field) are selected by lane masks e{i}: the lanes whose count
    is k (and whose spin is s). Lanes with P = 2^32 accept outright, lanes with 0 < P < 2^32 start undecided,
    and each level of U, most significant first, decides the lanes whose random bit differs from P's bit."""
    w(f"    tb = p_ptr + wv[None, :] * {NT} + ze[:, None] * {Z1}")
    J, h, Z = check(sys_)
    # Metropolis on a periodic, unmasked lattice: every site has z_eff = Z, so the sign of dE of each entry is
    # known here and the entries with dE <= 0 (p = 1, P = 2^32 in the table) accept without a load or a draw
    static = sys_.acceptance != "barker" and sys_.mask is None and all(sys_.periodic)
    w("    acc = tl.zeros([BLOCK, G], dtype=tl.int32)")
    ent = []           # (lane-mask expression, table offset)
    for k in range(Z1):
        terms = [f"c{p}" if (k >> p) & 1 else f"~c{p}" for p in range(NP)]
        w(f"    eq = {' & '.join(terms)}")
        for sel, off, sigma in ((["eq & ~s0", k, -1], ["eq & s0", Z1 * Z1 + k, 1]) if field else (["eq", k, -1],)):
            if static and k <= Z and 2.0 * J * (Z - 2 * k) + 2.0 * h * sigma <= 0.0:
                w(f"    acc = acc | ({sel})")
                continue
            w(f"    e{len(ent)} = {sel}")
            ent.append(off)
    w("    und = tl.zeros([BLOCK, G], dtype=tl.int32)")
    for i, off in enumerate(ent):
        if static:     # z_eff = Z everywhere: one table value per word, shape [1, G]
            w(f"    P{i} = tl.load(p_ptr + wv[None, :] * {NT} + {Z * Z1 + off}, mask=wok[None, :], other=0)")
        else:
            w(f"    P{i} = tl.load(tb + {off}, mask=ldw, other=0)")
        w(f"    acc = acc | tl.where(P{i} >= {2 ** 32}, e{i}, 0)")
        w(f"    und = und | tl.where((P{i} > 0) & (P{i} < {2 ** 32}), e{i}, 0)")
        # undecided lanes have P < 2^32: the level loop works on 32-bit words
        w(f"    P{i} = P{i}.to(tl.uint32)")
    # Philox counter word * nsite + site (int32, as in the shared mode), key per level group: grid-independent
    w(f"    ctr = (rbase[None, :] + own[:, None].to(tl.int64)).to(tl.int32)")

    def group(indent: str, gexpr: str):
        w(f"{indent}kg = ((seed.to(tl.int64) + ({gexpr} + 1) * {GROUP_KEY_STEP}) % 2147483629).to(tl.int32)")
        w(f"{indent}q0, q1, q2, q3 = tl.randint4x(kg, ctr, n_rounds={INDEPENDENT_ROUNDS})")
        for q in range(4):
            w(f"{indent}rw = q{q}.to(tl.int32, bitcast=True)")
            w(f"{indent}sh = ({LEVELS - 1 - q} - 4 * {gexpr}).to(tl.uint32)")
            # bit sh of each entry's P as an all-ones / all-zeros mask, then one AND per entry
            masks = " | ".join(f"(e{i} & (0 - ((P{i} >> sh) & 1)).to(tl.int32, bitcast=True))" for i in range(len(ent))) or "0"
            w(f"{indent}pb = {masks}")
            w(f"{indent}acc = acc | (und & ~rw & pb)")
            w(f"{indent}und = und & ~(rw ^ pb)")

    # the first two level groups (8 levels) run unconditionally: they decide all but about 2^-8 of the lanes,
    # and the tile-wide reduction is paid only once, for the rare tail; the stream and the result are unchanged
    if ent:
        group("    ", "(tl.program_id(0) * 0).to(tl.int64)")
        group("    ", "(tl.program_id(0) * 0 + 1).to(tl.int64)")
    w("    g = tl.program_id(0) * 0 + 2")
    w("    nund = tl.sum(tl.sum((und != 0).to(tl.int32), axis=1), axis=0)")
    w(f"    while (g < {LEVELS // 4}) & (nund > 0):")
    group("        ", "g.to(tl.int64)")
    w("        g = g + 1")
    w("        nund = tl.sum(tl.sum((und != 0).to(tl.int32), axis=1), axis=0)")
