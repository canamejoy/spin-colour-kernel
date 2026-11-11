"""Bitwise multi-spin-coded Ising (asynchronous multi-spin coding over replicas), on the Triton CPU interpreter.

The kernel updates 32 replicas per int32 word with bit operations: anti-aligned neighbour counts in bit
planes, one uniform per (word, site, sweep) shared by the word's 32 replicas, and a host-side acceptance
table p(s, z_eff, k). The exactness tests compare it, spin by spin, with an independent sequential chain on
unpacked +-1 spins that draws the same uniforms from a NumPy Philox and reads the same table.
"""

import math

import numpy as np
import pytest
import torch

from Spin_Colour_Kernel import bitwise as bw
from Spin_Colour_Kernel import crystals as cr
from Spin_Colour_Kernel import system as sy
from Spin_Colour_Kernel.sampler import Sampler


def _masked_square():
    rng = np.random.default_rng(0)
    mask = rng.random((1, 1, 8, 8)) > 0.25
    return sy.System("Ising square masked h", 1, cr.make("square"), (8, 8), (True, True), {1: 1.0},
                     h=(0.3,), mask=mask)


def _masked_open_sc():
    mask = np.ones((1, 3, 3, 4), bool)
    mask[0, 1, 1, 1:3] = False
    return sy.System("Ising sc open masked", 1, cr.make("sc"), (4, 3, 3), (False,) * 3, {1: 1.0}, mask=mask)


EXACT = [sy.ising("square", 8), sy.ising("triangular", 6), sy.ising("sc", 4), sy.ising("fcc", 4),
         _masked_square(), _masked_open_sc()]


def _with(s, acceptance):
    s.acceptance = acceptance
    return s


@pytest.mark.parametrize("acceptance", ["metropolis", "barker"])
@pytest.mark.parametrize("s", EXACT, ids=lambda s: f"{s.name} {s.box}")
def test_bitwise_kernel_equals_sequential_reference(s, acceptance):
    """B = 64 (two words at two temperatures): the kernel and the sequential per-replica chain with the same
    uniforms and the same table hold identical spins after every sweep."""
    s = _with(s, acceptance)
    A = Sampler(s, 64, device="cpu", seed=13, bitwise=True)
    assert A.bitwise and A.packed and A.W == 2
    Tc = s.T_ref or 3.0
    A.set_temperature(torch.tensor([2.0] * 32 + [4.0] * 32) * (Tc / 3.0))
    S = A.init_spins(seed=3)
    ref = A.unpack(S).numpy()[:, 0].copy()          # (B, b, L3, L2, L1) +-1, 0 on masked sites
    S0 = S.clone()
    for k in range(3):
        sweep = int(A.state[1])
        A.sweep(S)
        ref = bw.reference_sweep(s, ref, A.ptab.numpy(), int(A.state[0]), sweep)
        got = A.unpack(S).numpy()[:, 0]
        assert int((got != ref).sum()) == 0, f"sweep {k + 1}"
    assert not torch.equal(S, S0)
    m = torch.as_tensor(s.site_mask())
    assert torch.all(A.unpack(S)[:, 0][:, ~m] == 0)


@pytest.mark.parametrize("G", [1, 4, 8])
def test_bitwise_stream_does_not_depend_on_words_per_program(G):
    """W = 12 words (a partial last block when G = 8): one, four or eight words per program draw the same
    uniforms, and the kernel still equals the sequential reference."""
    s = _masked_square()
    A = Sampler(s, 32 * 12, device="cpu", seed=8, bitwise=True, words_per_program=G)
    assert A.G == G and A.ngroups == -(-12 // G)
    A.set_temperature(torch.arange(12, dtype=torch.float32).repeat_interleave(32) * 0.25 + 1.5)
    S = A.init_spins(seed=6)
    ref = A.unpack(S).numpy()[:, 0].copy()
    for _ in range(2):
        sweep = int(A.state[1])
        A.sweep(S)
        ref = bw.reference_sweep(s, ref, A.ptab.numpy(), int(A.state[0]), sweep)
        assert int((A.unpack(S).numpy()[:, 0] != ref).sum()) == 0


def test_bitwise_kernel_reference_launches_and_sync_path():
    """The per-site launch of the same kernel (Sampler.reference_sweep) and the compacted class launches agree;
    the synchronous control runs."""
    s = sy.System("Ising sc odd", 1, cr.make("sc"), (5, 5, 5), (True,) * 3, {1: 1.0})
    A, R = Sampler(s, 96, device="cpu", seed=2, bitwise=True), Sampler(s, 96, device="cpu", seed=2, bitwise=True)
    assert A.stride > 1 and A.W == 3
    for X in (A, R):
        X.set_temperature(4.5)
    S1 = A.init_spins(seed=1)
    S2 = S1.clone()
    A.sweep(S1, 2)
    R.reference_sweep(S2, 2)
    assert torch.equal(S1, S2)
    A.sync_sweep(S1, 1)


def test_numpy_philox_matches_triton_rand4x():
    import triton
    import triton.language as tl

    @triton.jit
    def k(out_ptr, seed, N: tl.constexpr):
        off = tl.arange(0, N)
        a, b, c, d = tl.rand4x(seed, off * 7 + 3)
        tl.store(out_ptr + off, a)
        tl.store(out_ptr + N + off, b)
        tl.store(out_ptr + 2 * N + off, c)
        tl.store(out_ptr + 3 * N + off, d)

    out = torch.zeros(4 * 64)
    k[(1,)](out, 123457, N=64)
    ref = bw.philox_uniform4(123457, np.arange(64) * 7 + 3)
    assert np.array_equal(out.numpy().reshape(4, 64), ref)


def test_acceptance_table_values():
    s = sy.ising("square", 4)
    beta = np.array([0.5])
    p = bw.p_table(s, beta)
    assert p.shape == (1, 2, 5, 5)
    # all four neighbours aligned (k = 0): dE = 2 J z = 8
    assert math.isclose(p[0, 1, 4, 0], math.exp(-4.0), rel_tol=1e-6)
    assert p[0, 1, 4, 2] == 1.0 and p[0, 1, 4, 4] == 1.0
    s.acceptance = "barker"
    p = bw.p_table(s, beta)
    assert math.isclose(p[0, 0, 4, 2], 0.5, rel_tol=1e-6)
    assert math.isclose(p[0, 0, 4, 0], 1 / (1 + math.exp(4.0)), rel_tol=1e-6)


@pytest.mark.parametrize("make, msg", [
    (lambda: sy.heisenberg("sc", 4), "Ising"),
    (lambda: sy.System("J1-J2", 1, cr.make("sc", (1, 2)), (4, 4, 4), (True,) * 3, {1: 1.0, 2: 0.5}), "uniform"),
    (lambda: sy.System("DMI", 1, cr.make("square"), (4, 4), (True,) * 2, {1: 1.0}, extra={0: np.eye(1)}), "extra"),
])
def test_bitwise_refuses_other_systems(make, msg):
    with pytest.raises(ValueError, match=msg):
        Sampler(make(), 32, device="cpu", bitwise=True)


def test_bitwise_needs_one_temperature_per_word():
    A = Sampler(sy.ising("square", 4), 64, device="cpu", bitwise=True)
    A.set_temperature(torch.tensor([2.0] * 32 + [3.0] * 32))
    with pytest.raises(ValueError, match="word"):
        A.set_temperature(torch.linspace(2.0, 3.0, 64))


def test_bitwise_barker_energy_histogram_matches_exact_enumeration():
    """Ising square 4x4 at Tc, Barker: one replica per word (replicas of a word share uniforms and may
    coalesce), thinned well beyond the autocorrelation time, against exact enumeration."""
    from scipy import stats

    from Spin_Colour_Kernel.exp_core import exact_energy_distribution
    s = sy.ising("square", 4)
    s.acceptance = "barker"
    T = 2.0 / math.log(1 + math.sqrt(2))
    W = 256                                         # one program covers every word: the interpreter cost is per program
    A = Sampler(s, 32 * W, device="cpu", seed=21, bitwise=True, words_per_program=W)
    A.set_temperature(T)
    S = A.init_spins(seed=4)
    A.sweep(S, 40)
    Es = []
    for _ in range(20):
        A.sweep(S, 8)
        x = A.unpack(S)[:: 32]                     # replica 0 of every word
        Es.append(A.energy(x).numpy())
    Es = np.round(np.concatenate(Es), 6)
    ex = exact_energy_distribution(s, T)
    lv = ex["levels"]
    idx = np.searchsorted(lv, Es)
    assert np.allclose(lv[np.clip(idx, 0, len(lv) - 1)], Es)
    obs = np.bincount(idx, minlength=len(lv)).astype(float)
    exp_ = ex["p"] * obs.sum()
    o2, e2, co, ce = [], [], 0.0, 0.0
    for o, e in zip(obs, exp_):
        co, ce = co + o, ce + e
        if ce >= 5:
            o2.append(co)
            e2.append(ce)
            co, ce = 0.0, 0.0
    o2[-1] += co
    e2[-1] += ce
    _, p = stats.chisquare(o2, e2)
    assert p > 0.01


# ------------------------------------------------------------------ independent mode: one draw per replica
TC_SQUARE = 2.0 / math.log(1 + math.sqrt(2))


def test_numpy_philox_raw_matches_triton_randint4x():
    """The integer twin reproduces tl.randint4x for int64 offsets whose high word is the Philox counter c1."""
    import triton
    import triton.language as tl

    @triton.jit
    def k(out_ptr, seed, N: tl.constexpr):
        off = tl.arange(0, N)
        o = (off * 7 + 3).to(tl.int64) + ((off % 5 + 1).to(tl.int64) << 32)
        a, b, c, d = tl.randint4x(seed, o)
        tl.store(out_ptr + off, a.to(tl.int32, bitcast=True))
        tl.store(out_ptr + N + off, b.to(tl.int32, bitcast=True))
        tl.store(out_ptr + 2 * N + off, c.to(tl.int32, bitcast=True))
        tl.store(out_ptr + 3 * N + off, d.to(tl.int32, bitcast=True))

    out = torch.zeros(4 * 64, dtype=torch.int32)
    k[(1,)](out, 123457, N=64)
    n = np.arange(64)
    ref = bw.philox_raw4(123457, n * 7 + 3, n % 5 + 1)
    assert ref.dtype == np.uint32
    assert np.array_equal(out.numpy().view(np.uint32).reshape(4, 64), ref)
    # the float twin is the same generator with c1 = 0
    r = bw.philox_raw4(99, n).view(np.int32)
    assert np.array_equal(bw.philox_uniform4(99, n), np.where(r < 0, ~r, r).astype(np.float32) * np.float32(4.6566127342e-10))


def test_fixed_point_table_quantizes_down_and_keeps_extremes():
    s = sy.ising("square", 4)
    P = bw.p_fixed(s, np.array([0.5]))
    p = bw.p_table(s, np.array([0.5])).astype(np.float64)
    assert P.dtype == np.int64 and P.shape == p.shape
    assert P[0, 1, 4, 2] == 2 ** 32                   # p = 1: always accept, U < 2^32 for every U
    assert P[0, 1, 4, 0] == math.floor(math.exp(-4.0) * 2 ** 32)
    assert np.all((P >= 0) & (P <= 2 ** 32))


@pytest.mark.parametrize("acceptance", ["metropolis", "barker"])
@pytest.mark.parametrize("s", EXACT, ids=lambda s: f"{s.name} {s.box}")
def test_independent_kernel_equals_sequential_reference(s, acceptance):
    """independent=True: every replica draws its own 32-bit uniform U, built bit by bit from the kernel's
    Philox words; the sequential reference compares U < P per replica as integers and must equal the
    bit-sliced kernel spin for spin after every sweep (B = 64, two words at two temperatures)."""
    s = _with(s, acceptance)
    A = Sampler(s, 64, device="cpu", seed=13, bitwise=True, independent=True)
    assert A.bitwise and A.independent and A.W == 2
    Tc = s.T_ref or 3.0
    A.set_temperature(torch.tensor([2.0] * 32 + [4.0] * 32) * (Tc / 3.0))
    assert A.ptab.dtype == torch.int64
    S = A.init_spins(seed=3)
    ref = A.unpack(S).numpy()[:, 0].copy()
    S0 = S.clone()
    for k in range(3):
        sweep = int(A.state[1])
        A.sweep(S)
        ref = bw.reference_sweep(s, ref, A.ptab.numpy(), int(A.state[0]), sweep, independent=True)
        got = A.unpack(S).numpy()[:, 0]
        assert int((got != ref).sum()) == 0, f"sweep {k + 1}"
    assert not torch.equal(S, S0)
    m = torch.as_tensor(s.site_mask())
    assert torch.all(A.unpack(S)[:, 0][:, ~m] == 0)


@pytest.mark.parametrize("G", [1, 8])
def test_independent_stream_does_not_depend_on_words_per_program(G):
    s = _masked_square()
    A = Sampler(s, 32 * 12, device="cpu", seed=8, bitwise=True, independent=True, words_per_program=G)
    A.set_temperature(torch.arange(12, dtype=torch.float32).repeat_interleave(32) * 0.25 + 1.5)
    S = A.init_spins(seed=6)
    ref = A.unpack(S).numpy()[:, 0].copy()
    sweep = int(A.state[1])
    A.sweep(S)
    ref = bw.reference_sweep(s, ref, A.ptab.numpy(), int(A.state[0]), sweep, independent=True)
    assert int((A.unpack(S).numpy()[:, 0] != ref).sum()) == 0


def test_shared_mode_source_unchanged_by_the_independent_option():
    from Spin_Colour_Kernel import generator as gen
    s = sy.ising("square", 8)
    assert gen.source(s, True, True) == bw.source(s) == gen.source(s, True, True, independent=False)
    assert gen.source(s, True, True, independent=True) != gen.source(s, True, True)
    with pytest.raises(ValueError, match="bitwise"):
        Sampler(s, 32, device="cpu", independent=True)


def _distinct(A, S):
    x = A.unpack(S)[:, 0].reshape(A.B, -1).numpy()
    return len(np.unique(x, axis=0))


def _square_tc(independent, seed=5):
    s = sy.ising("square", 8)
    s.acceptance = "barker"
    A = Sampler(s, 32, device="cpu", seed=seed, bitwise=True, independent=independent)
    A.set_temperature(TC_SQUARE)
    return A


def test_independent_replicas_do_not_coalesce(monkeypatch):
    """Ising square 8x8 at Tc, Barker, one word. (a) From 32 identical configurations the shared draw keeps
    one configuration and the independent draws give at least 30 after 20 sweeps. (b) From random starts the
    shared draw collapses the word to a few configurations within 400 sweeps; independent replicas share a
    configuration only when it is fully ordered, which independent chains reach by chance on 8x8 at Tc. The draws do not depend on the launch grid, so the test compacts the checkerboard
    (stride 2) into one program per class launch: the interpreter's cost is per program."""
    import Spin_Colour_Kernel.sampler as sm
    monkeypatch.setattr(sm, "COMPACT_MIN", 2)
    res = {}
    for ind in (False, True):
        A = _square_tc(ind)
        assert A.stride == 2 and A.rows_per == 8
        x = A.unpack(A.init_spins(seed=2))
        S = A.pack(x[:1].expand_as(x).float())
        assert _distinct(A, S) == 1
        A.sweep(S, 20)
        res[ind, "same"] = _distinct(A, S)
        A = _square_tc(ind)
        S = A.init_spins(seed=2)
        assert _distinct(A, S) == 32
        A.sweep(S, 400)
        res[ind, "random"] = _distinct(A, S)
        x = A.unpack(S).numpy().reshape(32, -1)
        rows, counts = np.unique(x, axis=0, return_counts=True)
        # configurations shared by several replicas that are not fully ordered: coalescence, not chance (an
        # 8x8 lattice at Tc is fully ordered often enough for independent replicas to meet there)
        res[ind, "repeated_disordered"] = int(sum(1 for r, c in zip(rows, counts) if c > 1 and abs(r.mean()) < 1))
    assert res[False, "same"] == 1 and res[True, "same"] >= 30, res
    assert res[False, "random"] <= 4 and res[True, "repeated_disordered"] == 0 and res[True, "random"] >= 28, res


def test_independent_barker_energy_histogram_all_replicas_match_exact_enumeration():
    """Ising square 4x4 at Tc, Barker, independent draws: the energies of ALL replicas (no per-word
    subsampling) against exact enumeration. Seeds fixed before any p-value was seen."""
    from scipy import stats

    from Spin_Colour_Kernel.exp_core import exact_energy_distribution
    s = sy.ising("square", 4)
    s.acceptance = "barker"
    W = 16
    A = Sampler(s, 32 * W, device="cpu", seed=31, bitwise=True, independent=True, words_per_program=W)
    A.set_temperature(TC_SQUARE)
    S = A.init_spins(seed=7)
    A.sweep(S, 40)
    Es = []
    for _ in range(12):
        A.sweep(S, 8)
        Es.append(A.energy(A.unpack(S)).numpy())
    Es = np.round(np.concatenate(Es), 6)
    ex = exact_energy_distribution(s, TC_SQUARE)
    lv = ex["levels"]
    idx = np.searchsorted(lv, Es)
    assert np.allclose(lv[np.clip(idx, 0, len(lv) - 1)], Es)
    obs = np.bincount(idx, minlength=len(lv)).astype(float)
    exp_ = ex["p"] * obs.sum()
    o2, e2, co, ce = [], [], 0.0, 0.0
    for o, e in zip(obs, exp_):
        co, ce = co + o, ce + e
        if ce >= 5:
            o2.append(co)
            e2.append(ce)
            co, ce = 0.0, 0.0
    o2[-1] += co
    e2[-1] += ce
    _, p = stats.chisquare(o2, e2)
    assert p > 0.01
