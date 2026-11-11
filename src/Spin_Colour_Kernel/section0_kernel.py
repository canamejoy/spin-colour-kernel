"""Section0 hand-specialized substep kernel, vendored verbatim for experiment E7.

Source: github.com/canamejoy/Spin-State-GenerationAndInference, section0-mcmc.gpu/codigo/triton_halo.py
(commit fe8b65f), lines 60-233: the `_substep_kernel` only; its JAX plumbing is replaced by a torch
launcher in `exp_perf.section0_launcher`. Same author as this repository.
"""

import triton
import triton.language as tl

@triton.jit
def _substep_kernel(
    sp_ptr,          # (B, 3, LZ, N, N)  f32   spins in
    k1_ptr,          # (B, LZ, N, N)     f32   K1_eff, surface-dependent
    par_ptr,         # (B, 6)            f32   Jex, Kan2, Hex, KDM, cos g, sin g
    disk_ptr,        # (N, N)            f32   1.0 inside the disk
    beta_ptr,        # (1,)              f32
    step_ptr,        # (1,)              i32   mixes into the Philox seed
    tr_ptr,          # (B, 3, LZ, N, N)  f32   trial spins   (HAS_RNG=False)
    ru_ptr,          # (B, LZ, N, N)     f32   uniforms      (HAS_RNG=False)
    out_ptr,         # (B, 3, LZ, N, N)  f32   spins out
    de_ptr,          # (B, LZ, N, N)     f32   dE out        (WRITE_DE=True)
    N: tl.constexpr,
    LZ: tl.constexpr,
    PARITY: tl.constexpr,     # 0 updates the even sublattice, 1 the odd
    BY: tl.constexpr,         # rows of the block
    BX: tl.constexpr,         # lanes along x, a power of two >= N
    HAS_RNG: tl.constexpr,
    WRITE_DE: tl.constexpr,
    SO: tl.constexpr,
):
    b = tl.program_id(0)
    z = tl.program_id(1)
    ty = tl.program_id(2)

    NN = N * N
    sb = 3 * LZ * NN          # stride between replicas
    sc = LZ * NN              # stride between components

    # ---- the block, and its neighbour index vectors.
    # These are VECTORS over the block, not per-lane addresses: `xc + 1` is a
    # contiguous run shifted by one element, which is what keeps the loads
    # coalesced. That is the single difference from the gather attempts.
    ys = ty * BY + tl.arange(0, BY)
    xs = tl.arange(0, BX)
    live = (ys < N)[:, None] & (xs < N)[None, :]

    yc = tl.where(ys < N, ys, 0)
    xc = tl.where(xs < N, xs, 0)

    # x and y wrap; z does not (open boundary, handled by the gates below).
    ypi = tl.where(yc + 1 >= N, 0, yc + 1)
    ymi = tl.where(yc - 1 < 0, N - 1, yc - 1)
    xpi = tl.where(xc + 1 >= N, 0, xc + 1)
    xmi = tl.where(xc - 1 < 0, N - 1, xc - 1)

    o_c = yc[:, None] * N + xc[None, :]
    o_yp = ypi[:, None] * N + xc[None, :]
    o_ym = ymi[:, None] * N + xc[None, :]
    o_xp = yc[:, None] * N + xpi[None, :]
    o_xm = yc[:, None] * N + xmi[None, :]

    zp = tl.minimum(z + 1, LZ - 1)
    zm = tl.maximum(z - 1, 0)
    # `jnp.roll(...).at[:, -1].set(0.0)` in the reference: the plane past the
    # last layer contributes zero rather than wrapping.
    gp = tl.where(z < LZ - 1, 1.0, 0.0)
    gm = tl.where(z > 0, 1.0, 0.0)

    # ---- the site mask: inside the disk, and on the sublattice being updated.
    disk = tl.load(disk_ptr + o_c, mask=live, other=0.0)
    par = (xc[None, :] + yc[:, None] + z) % 2
    upd = live & (disk > 0.5) & (par == PARITY)

    # ---- per-chain parameters.
    Jex = tl.load(par_ptr + b * 6 + 0)
    Kan2 = tl.load(par_ptr + b * 6 + 1)
    Hex = tl.load(par_ptr + b * 6 + 2)
    KDM = tl.load(par_ptr + b * 6 + 3)
    cosg = tl.load(par_ptr + b * 6 + 4)
    sing = tl.load(par_ptr + b * 6 + 5)
    beta = tl.load(beta_ptr)

    sca = sp_ptr + b * sb
    k1 = tl.load(k1_ptr + b * LZ * NN + z * NN + o_c, mask=live, other=0.0)

    # ---- the spin at the site.
    s0 = tl.load(sca + 0 * sc + z * NN + o_c, mask=live, other=0.0)
    s1 = tl.load(sca + 1 * sc + z * NN + o_c, mask=live, other=0.0)
    s2 = tl.load(sca + 2 * sc + z * NN + o_c, mask=live, other=0.0)

    # ---- the trial spin: a uniform point on the sphere.
    if HAS_RNG:
        off = ((b * LZ + z) * N + yc[:, None]) * N + xc[None, :]
        # One key, four distinct COUNTERS -- the textbook use of a
        # counter-based generator. An earlier version varied the KEY instead
        # (`sd+0..3` at a shared counter) and the engine then accepted 1.1%
        # less than production, systematically across five temperatures,
        # while its proposals were provably isotropic and unit-norm. That
        # leaves the uniform: adjacent keys at one counter let `ru` correlate
        # with the proposed direction, hence with dE, which biases the
        # Metropolis test. Separating the counters removes the shared input.
        sd = tl.load(step_ptr)
        o4 = off * 4
        g0 = tl.randn(sd, o4 + 0)
        g1 = tl.randn(sd, o4 + 1)
        g2 = tl.randn(sd, o4 + 2)
        nrm = tl.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-30
        t0 = g0 / nrm * SO
        t1 = g1 / nrm * SO
        t2 = g2 / nrm * SO
        ru = tl.rand(sd, o4 + 3)
    else:
        trb = tr_ptr + b * sb
        t0 = tl.load(trb + 0 * sc + z * NN + o_c, mask=live, other=0.0)
        t1 = tl.load(trb + 1 * sc + z * NN + o_c, mask=live, other=0.0)
        t2 = tl.load(trb + 2 * sc + z * NN + o_c, mask=live, other=0.0)
        ru = tl.load(ru_ptr + b * LZ * NN + z * NN + o_c, mask=live, other=0.0)

    d0 = t0 - s0
    d1 = t1 - s1
    d2 = t2 - s2

    # ---- the stencil, one component at a time so the six neighbour blocks of
    # a component die before the next component's are loaded. Holding all 21
    # at once spills at BY >= 8.
    p0 = sca + 0 * sc
    n_xp = tl.load(p0 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p0 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p0 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p0 + z * NN + o_ym, mask=live, other=0.0)
    sv0 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p0 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p0 + zm * NN + o_c, mask=live, other=0.0))
    dyx = n_yp - n_ym                     # DMI needs this difference only

    p1 = sca + 1 * sc
    n_xp = tl.load(p1 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p1 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p1 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p1 + z * NN + o_ym, mask=live, other=0.0)
    sv1 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p1 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p1 + zm * NN + o_c, mask=live, other=0.0))
    dxy = n_xp - n_xm

    p2 = sca + 2 * sc
    n_xp = tl.load(p2 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p2 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p2 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p2 + z * NN + o_ym, mask=live, other=0.0)
    sv2 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p2 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p2 + zm * NN + o_c, mask=live, other=0.0))
    dxz = n_xp - n_xm
    dyz = n_yp - n_ym

    # ---- the energy difference, term for term as in `sim_core.substep`.
    dE = -Jex * (d0 * sv0 + d1 * sv1 + d2 * sv2)

    i4 = 1.0 / (SO * SO * SO * SO)
    i6 = i4 / (SO * SO)
    an_t = (k1 * i4 * ((t0 * t1) * (t0 * t1) + (t0 * t2) * (t0 * t2)
                       + (t1 * t2) * (t1 * t2))
            + Kan2 * i6 * (t0 * t1 * t2) * (t0 * t1 * t2))
    an_s = (k1 * i4 * ((s0 * s1) * (s0 * s1) + (s0 * s2) * (s0 * s2)
                       + (s1 * s2) * (s1 * s2))
            + Kan2 * i6 * (s0 * s1 * s2) * (s0 * s1 * s2))
    dE += an_t - an_s

    # Interfacial DMI: in-plane bonds only, with the minus sign.
    dE -= KDM * (d1 * dxz - d2 * dxy + d2 * dyx - d0 * dyz)
    dE -= Hex * (d0 * cosg + d2 * sing)

    acc = ((dE < 0.0) | (ru < tl.exp(-dE * beta))) & upd

    ob = out_ptr + b * sb + z * NN + o_c
    tl.store(ob + 0 * sc, tl.where(acc, t0, s0), mask=live)
    tl.store(ob + 1 * sc, tl.where(acc, t1, s1), mask=live)
    tl.store(ob + 2 * sc, tl.where(acc, t2, s2), mask=live)
    if WRITE_DE:
        tl.store(de_ptr + b * LZ * NN + z * NN + o_c, dE, mask=live)


