"""The kernel generator (proposal r07, section 5): one System -> one specialized Triton kernel.

Spins are read in the storage type of `src_ptr` (int8 for Ising, float32 for vector models) and
converted to float32 for the arithmetic; results are stored back in the type of `dst_ptr`.

The model (n, proposal, anisotropy), the bond table, the colouring and the box are emitted as
straight-line Triton source; only the replica index, the inverse temperature and the counter state
are runtime values. One launch updates one (colour, sublattice) class; a sweep is the composition
of the n_c * b class launches, in a fixed order.

Colour compaction. Along x a linear colouring repeats with period P = lcm_r m_r / gcd(K_r1, m_r)
(STRIDE = P), and along a row each colour of the row's coset appears at most once per period. A class
launch therefore gives each program a tile of R rows x BX = ceil(L1 / P) sites: each lane solves its
row's residue x0 from the colouring and takes x0 + P * j. A row without the colour does nothing, and
when P is close to L1 one program still covers R rows instead of a single site. Every load of a class launch is masked by the update predicate, so a sweep reads the
lattice about once whatever n_c is, instead of n_c times. The sequential reference and the synchronous control keep STRIDE = 1, R = 1.

Randomness is Philox (`tl.rand4x`) keyed by a per-sweep seed and countered by (replica, site), so the
stream is a pure function of the site and the sweep, never of the launch grid. The same compiled
kernel serves the parallel chain and the sequential reference: the reference launches it once per
site (`x_only`, `row0`), the parallel chain once per class.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

from .system import System

CACHE = Path(os.environ.get("SCK_CACHE", Path.home() / ".cache" / "spin_colour_kernel"))


def _f(x: float) -> str:
    return repr(float(np.float32(x)))


def _coord_expr(var: str, d: int, L: int, periodic: bool) -> tuple[str, str | None]:
    """Neighbour coordinate and its validity predicate (None when always valid)."""
    if d == 0:
        return var, None
    if periodic:
        return f"(({var} + {d % L}) % {L})", None
    e = f"({var} + {d})"
    return e, f"(({e} >= 0) & ({e} < {L}))"


def x_period(col) -> int:
    """Period along x of a linear colouring: lcm over its factors of m_r / gcd(K_r1, m_r)."""
    P = 1
    for r, mr in enumerate(col.m):
        P = math.lcm(P, mr // math.gcd(int(col.K[r][0]) % mr or mr, mr))
    return P


def _colour_expr(sys_: System, xv: str, a: int) -> str:
    """Colour of site (xv, y, z) of sublattice a, as a Triton expression (mixed-radix, eq. 6)."""
    col, D = sys_.colouring, sys_.crystal.D
    if not col.m:
        return f"{xv} * 0"
    terms = []
    for r, mr in enumerate(col.m):
        coeffs = list(col.K[r]) + [0] * (3 - D)
        terms.append((f"({coeffs[0]} * {xv} + {coeffs[1]} * y + {coeffs[2]} * z + {col.o[a][r]}) % {mr}", mr))
    expr = terms[0][0]
    for e, mr in terms[1:]:
        expr = f"({expr}) * {mr} + {e}"
    return expr


def _emit_tile(w, sys_: System, chunked: bool) -> None:
    """Lanes, rows and the colour-compacted x coordinate of one class launch; defines live, xc, yv, zv.

    chunked: one-row programs take their x chunk from grid axis 2 (rows longer than one program's lanes)."""
    b, L1, L2, L3 = sys_.b, sys_.L1, sys_.L2, sys_.L3
    # a program covers a tile of R rows x BX sites, flattened onto BLOCK lanes: lane l is row l // BX
    # of the tile and the (l % BX)-th site of that row's colour class (x0 + STRIDE * j)
    w("    lane = tl.arange(0, BLOCK)")
    w("    if R == 1:")                       # one row per program: row coordinates stay scalars
    w("        j = lane + tl.program_id(2) * BLOCK" if chunked else "        j = lane")
    w("        row = tl.program_id(1) + row0")
    w(f"        z = row // {L2}")
    w(f"        y = row % {L2}")
    w("    else:")
    w("        j = lane % BX")
    w("        rowv = tl.program_id(1) * R + lane // BX + row0")
    w(f"        rok = (lane < R * BX) & (rowv < {L3 * L2})")
    w("        rowc = tl.where(rok, rowv, 0)")
    w(f"        z = rowc // {L2}")
    w(f"        y = rowc % {L2}")

    # compaction: along a row the colour has period STRIDE and each colour of the row's coset appears
    # at most once per period, so each lane finds its row's residue x0 (L1 when the row lacks it)
    w("    if STRIDE > 1:")
    w(f"        x0 = tl.zeros([BLOCK], dtype=tl.int32) + {L1}")
    w("        for q in tl.static_range(STRIDE):")
    for a in range(b):
        kw = "if" if a == 0 else "elif"
        w(f"            {kw} ALPHA == {a}:")
        w(f"                cq = {_colour_expr(sys_, 'q', a)}")
        w("                x0 = tl.where(cq == COLOUR, q, x0)")
    w("    else:")
    w("        x0 = 0")
    w("    x = x0 + STRIDE * j")
    w("    xo = tl.zeros([BLOCK], dtype=tl.int32) + x_only")
    w("    if R == 1:")
    w(f"        live = (x < {L1}) & ((x == xo) | (xo < 0))")
    w("    else:")
    w(f"        live = rok & (x < {L1}) & ((x == xo) | (xo < 0))")
    w("    xc = tl.where(live, x, 0)")
    w("    yv = tl.zeros([BLOCK], dtype=tl.int32) + y")
    w("    zv = tl.zeros([BLOCK], dtype=tl.int32) + z")


def _emit_class(w, sys_: System) -> None:
    """Update predicate of one (colour, sublattice) class: defines colour, upd, own, m_own (masked systems), ld."""
    b, L1, L2 = sys_.b, sys_.L1, sys_.L2
    pl = sys_.L3 * L2 * L1
    has_mask = sys_.mask is not None
    # colour mask, from (6), per alpha
    for a in range(b):
        kw = "if" if a == 0 else "elif"
        w(f"    {kw} ALPHA == {a}:")
        w(f"        colour = {_colour_expr(sys_, 'xc', a)}")
    w("    if COLOUR < 0:")
    w("        upd = live")
    w("    else:")
    w("        upd = live & (colour == COLOUR)")
    w(f"    own = ALPHA * {pl} + z * {L2 * L1} + y * {L1} + xc")
    if has_mask:
        w("    m_own = tl.load(mask_ptr + own, mask=live, other=0)")
        w("    upd = upd & (m_own != 0)")
    # a class launch reads only the sites it updates; the synchronous control reads every live site
    w("    if SYNC:")
    w("        ld = live")
    w("    else:")
    w("        ld = upd")


def source(sys_: System, packed: bool = False, bitwise: bool = False, independent: bool = False) -> str:
    """Triton source text for one system.

    bitwise (Ising only): true multi-spin coding, see `bitwise.source`; same packed storage, but the
    update stays in bit operations with one uniform per (word, site, sweep) and a host acceptance table.
    independent (bitwise only): one 32-bit uniform per replica instead, compared bit-sliced.

    packed (Ising only): multi-spin coding over replicas. Word w of a site holds that site's spin in
    replicas 32 w .. 32 w + 31, one bit each (1 = +1). A program loads one word per neighbour, unpacks it
    into a [BLOCK, 32] tile, runs the same Metropolis step with the same Philox counter per (replica,
    site) as the unpacked kernel, and packs the result back with an OR. Rows longer than BLOCK are split
    over grid axis 2."""
    if independent and not bitwise:
        raise ValueError("independent per-replica draws need the bitwise kernel")
    if bitwise:
        from .bitwise import source as bitwise_source
        return bitwise_source(sys_, independent)
    n, b = sys_.n, sys_.b
    if packed and n != 1:
        raise ValueError("bit packing is for Ising (n = 1) only")
    P = packed
    RB = "BLOCK, 32" if P else "BLOCK"
    L1, L2, L3 = sys_.L1, sys_.L2, sys_.L3
    D = sys_.crystal.D
    per = list(sys_.periodic) + [False] * (3 - D)
    pl = L3 * L2 * L1                      # one (component, alpha) plane
    nsite = b * pl
    has_mask = sys_.mask is not None
    has_cubic = sys_.cubic_k1 is not None
    lines: list[str] = []
    w = lines.append
    w("import triton")
    w("import triton.language as tl")
    w("")
    w("")
    if P:
        w("@triton.jit")
        w("def _or(a, b):")
        w("    return a | b")
        w("")
        w("")
    w('@triton.jit(do_not_specialize=["x_only", "row0"])')
    w("def kernel(src_ptr, dst_ptr, mask_ptr, k1_ptr, beta_ptr, state_ptr, x_only, row0,")
    w("           ALPHA: tl.constexpr, COLOUR: tl.constexpr, BLOCK: tl.constexpr, SYNC: tl.constexpr,")
    w("           STRIDE: tl.constexpr, BX: tl.constexpr, R: tl.constexpr):")
    if P:
        w("    wrd = tl.program_id(0)")
        w("    rr = tl.arange(0, 32)")
        w("    rep = wrd * 32 + rr")
    else:
        w("    rep = tl.program_id(0)")
    _emit_tile(w, sys_, True)     # one-row programs may be chunked over grid axis 2 (packed: PTILE; float: MAX_BLOCK)
    w("    beta = tl.load(beta_ptr + rep)")
    w("    base_seed = tl.load(state_ptr + 0)")
    w("    sweep = tl.load(state_ptr + 1)")
    w("    seed = ((base_seed * 1000003 + sweep * 7919 + 12345) % 2147483629).to(tl.int32)")
    w(f"    rbase = wrd.to(tl.int64) * {nsite}" if P else f"    rbase = rep.to(tl.int64) * {n * nsite}")
    _emit_class(w, sys_)
    # own spin
    if P:
        w("    w0 = tl.load(src_ptr + rbase + own, mask=ld, other=0)")
        w("    s0 = (((w0[:, None] >> rr[None, :]) & 1) * 2 - 1).to(tl.float32)")
    else:
        for k in range(n):
            w(f"    s{k} = tl.load(src_ptr + rbase + {k * nsite} + own, mask=ld, other=0).to(tl.float32)")
    # local field, per alpha
    for a in range(b):
        kw = "if" if a == 0 else "elif"
        w(f"    {kw} ALPHA == {a}:")
        groups: dict[float, list[int]] = {}
        extras: list[int] = []
        for i, bd in enumerate(sys_.crystal.bonds):
            if bd.alpha != a:
                continue
            groups.setdefault(float(sys_.J.get(bd.shell, 0.0)), []).append(i)
            if i in sys_.extra:
                extras.append(i)
        loaded = {}
        for k in range(n):
            w(f"        v{k} = tl.zeros([{RB}], dtype=tl.float32)")
        idx = 0
        for Jv, members in groups.items():
            if Jv == 0.0 and not any(i in sys_.extra for i in members):
                continue
            for k in range(n):
                w(f"        g{k} = tl.zeros([{RB}], dtype=tl.float32)")
            for i in members:
                bd = sys_.crystal.bonds[i]
                d = list(bd.delta) + [0] * (3 - D)
                ex, vx = _coord_expr("xc", d[0], L1, per[0])
                ey, vy = _coord_expr("yv", d[1], L2, per[1])
                ez, vz = _coord_expr("zv", d[2], L3, per[2])
                valid = " & ".join(["ld"] + [p for p in (vx, vy, vz) if p])
                w(f"        q = {bd.alpha2 * pl} + {ez} * {L2 * L1} + {ey} * {L1} + {ex}")
                w(f"        ok = {valid}")
                w("        q = tl.where(ok, q, 0)")
                if P and has_mask:      # a masked neighbour holds no spin: it contributes 0, as in the unpacked store
                    w("        ok = ok & (tl.load(mask_ptr + q, mask=ok, other=0) != 0)")
                for k in range(n):
                    if P:
                        w("        tw = tl.load(src_ptr + rbase + q, mask=ok, other=0)")
                        w("        t = tl.where(ok[:, None], (((tw[:, None] >> rr[None, :]) & 1) * 2 - 1).to(tl.float32), 0.0)")
                    else:
                        w(f"        t = tl.load(src_ptr + rbase + {k * nsite} + q, mask=ok, other=0).to(tl.float32)")
                    w(f"        g{k} = g{k} + t")
                    if i in sys_.extra:
                        w(f"        n{idx}_{k} = t")
                if i in sys_.extra:
                    loaded[i] = idx
                    idx += 1
            if Jv != 0.0:
                for k in range(n):
                    w(f"        v{k} = v{k} + {_f(Jv)} * g{k}")
        for i in extras:
            E = sys_.extra[i]
            j = loaded[i]
            for k in range(n):
                terms = [f"{_f(E[k, l])} * n{j}_{l}" for l in range(n) if E[k, l] != 0.0]
                if terms:
                    w(f"        v{k} = v{k} + ({' + '.join(terms)})")
    if sys_.h is not None:
        for k in range(n):
            if sys_.h[k] != 0.0:
                w(f"    v{k} = v{k} + {_f(sys_.h[k])}")
    # proposal (9)
    w(f"    off = (rep[None, :] * {nsite} + own[:, None]).to(tl.int32)" if P else f"    off = (rep * {nsite} + own).to(tl.int32)")
    w("    u0, u1, u2, u3 = tl.rand4x(seed, off)")
    if n == 1:
        w("    t0 = -s0")
    elif n == 2:
        w("    phi = 6.283185307179586 * u1")
        w("    t0 = tl.cos(phi)")
        w("    t1 = tl.sin(phi)")
    else:
        w("    cz = 2.0 * u1 - 1.0")
        w("    phi = 6.283185307179586 * u2")
        w("    rr = tl.sqrt(tl.maximum(1.0 - cz * cz, 0.0))")
        w("    t0 = rr * tl.cos(phi)")
        w("    t1 = rr * tl.sin(phi)")
        w("    t2 = cz")
    # energy change (3)
    dE = " + ".join(f"(t{k} - s{k}) * v{k}" for k in range(n))
    w(f"    dE = -({dE})")
    if has_cubic:
        w("    k1 = tl.load(k1_ptr + own, mask=ld, other=0.0)")
        k2 = _f(sys_.cubic_k2)
        for p in ("t", "s"):
            w(f"    a_{p} = k1 * (({p}0 * {p}1) * ({p}0 * {p}1) + ({p}0 * {p}2) * ({p}0 * {p}2) "
              f"+ ({p}1 * {p}2) * ({p}1 * {p}2)) + {k2} * ({p}0 * {p}1 * {p}2) * ({p}0 * {p}1 * {p}2)")
        w("    dE = dE + (a_t - a_s)")
    bt, ud = ("beta[None, :]", "upd[:, None]") if P else ("beta", "upd")
    if sys_.acceptance == "barker":
        # heat-bath (Barker) rule: accept with 1 / (1 + exp(beta dE)); dE = 0 is accepted with probability 1/2
        w(f"    acc = (u0 * (1.0 + tl.exp({bt} * dE)) < 1.0) & {ud}")
    else:
        w(f"    acc = (u0 < tl.exp(-{bt} * dE)) & {ud}")
    if P:
        w("    up = (tl.where(acc, t0, s0) > 0).to(tl.int32) << rr[None, :]")
        w("    r0 = tl.reduce(up, 1, _or)")
    else:
        for k in range(n):
            w(f"    r{k} = tl.where(acc, t{k}, s{k}).to(dst_ptr.dtype.element_ty)")
    w("    if SYNC:")
    wm = "live & (m_own != 0)" if has_mask else "live"
    for k in range(n):
        w(f"        tl.store(dst_ptr + rbase + {k * nsite} + own, r{k}, mask={wm})")
    w("    else:")
    for k in range(n):
        w(f"        tl.store(dst_ptr + rbase + {k * nsite} + own, r{k}, mask=upd)")
    return "\n".join(lines) + "\n"


def compile_system(sys_: System, packed: bool = False, bitwise: bool = False, independent: bool = False):
    """Write the generated source to the cache, import it, and return the jitted kernel."""
    src = source(sys_, packed, bitwise, independent)
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"sck_{key}.py"
    if not path.exists():
        path.write_text(src)
    name = f"sck_{key}"
    if name in sys.modules:
        return sys.modules[name].kernel, src
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.kernel, src
