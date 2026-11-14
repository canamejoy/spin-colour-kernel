"""A system is the triple (spin model, crystal, finite geometry) with its couplings (proposal r07, eqs. 1-4).

Spins live in the component-planar, basis-planar layout of eq. (10):
    S[B, n, b, L3, L2, L1], fastest axis L1 (x). For D = 2, L3 = 1.
Masked sites hold the zero vector and are never updated, so a masked or open-boundary neighbour
contributes zero to the local field (3) without a mask load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import crystals as cr
from .colouring import Colouring, search, seam_ok, verify_on_box


@dataclass
class System:
    name: str
    n: int                                   # spin components: 1 Ising, 2 XY, 3 Heisenberg
    crystal: cr.Crystal
    box: tuple[int, ...]                     # (L1, L2[, L3]) cells
    periodic: tuple[bool, ...]
    J: dict[int, float]                      # isotropic exchange per shell
    extra: dict[int, np.ndarray] = field(default_factory=dict)   # bond index -> extra n x n matrix (DMI ...)
    h: tuple[float, ...] | None = None       # uniform field
    cubic_k1: np.ndarray | None = None       # per-site cubic anisotropy K1, shape (b, L3, L2, L1)
    cubic_k2: float = 0.0                    # (sx sy sz)^2 coefficient
    mask: np.ndarray | None = None           # bool (b, L3, L2, L1); False = no spin
    colouring: Colouring | None = None
    T_ref: float | None = None               # reference critical temperature, when one exists
    acceptance: str = "metropolis"           # "metropolis": u < exp(-beta dE); "barker": heat-bath rule

    def __post_init__(self):
        if self.n not in (1, 2, 3):
            raise ValueError("n must be 1, 2 or 3")
        if len(self.box) != self.crystal.D or len(self.periodic) != self.crystal.D:
            raise ValueError("box and periodic must have one entry per lattice axis")
        if self.cubic_k1 is not None and self.n != 3:
            raise ValueError("cubic anisotropy needs n = 3")
        if self.acceptance not in ("metropolis", "barker"):
            raise ValueError("acceptance must be 'metropolis' or 'barker'")
        if self.colouring is None:
            self.colouring = search(self.crystal, self.box, self.periodic)
            if self.colouring is None:
                raise ValueError(f"{self.name}: no seam-compatible linear colouring for box {self.box}")
        if not seam_ok(self.colouring, self.box, self.periodic):
            raise ValueError(f"{self.name}: colouring violates the seam condition (8) on box {self.box}")
        if not verify_on_box(self.colouring, self.crystal, self.box, self.periodic):
            raise ValueError(f"{self.name}: colouring is not admissible on the box")

    # ---------------------------------------------------------------- geometry
    @property
    def L1(self) -> int:
        return int(self.box[0])

    @property
    def L2(self) -> int:
        return int(self.box[1])

    @property
    def L3(self) -> int:
        return int(self.box[2]) if self.crystal.D == 3 else 1

    @property
    def b(self) -> int:
        return self.crystal.b

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.b, self.L3, self.L2, self.L1)

    def site_mask(self) -> np.ndarray:
        return np.ones(self.shape, bool) if self.mask is None else self.mask.astype(bool)

    @property
    def n_active(self) -> int:
        return int(self.site_mask().sum())

    def bond_matrix(self, i: int) -> np.ndarray:
        bd = self.crystal.bonds[i]
        M = self.J.get(bd.shell, 0.0) * np.eye(self.n)
        if i in self.extra:
            M = M + self.extra[i]
        return M

    def colour_map(self) -> np.ndarray:
        """Colour of every site, shape (b, L3, L2, L1)."""
        out = np.zeros(self.shape, np.int64)
        for a in range(self.b):
            for z in range(self.L3):
                for y in range(self.L2):
                    for x in range(self.L1):
                        c = (x, y, z)[: self.crystal.D]
                        out[a, z, y, x] = self.colouring.colour(c, a)
        return out


# -------------------------------------------------------------------- matrix
def ising(crystal: str, L: int, J: float = 1.0, shells=(1,), T_ref=None) -> System:
    c = cr.make(crystal, tuple(shells))
    return System(f"Ising {crystal}", 1, c, (L,) * c.D, (True,) * c.D, {s: J for s in shells}, T_ref=T_ref)


def xy(crystal: str, L: int, J: float = 1.0, T_ref=None) -> System:
    c = cr.make(crystal)
    return System(f"XY {crystal}", 2, c, (L,) * c.D, (True,) * c.D, {1: J}, T_ref=T_ref)


def heisenberg(crystal: str, L: int, J: float = 1.0, T_ref=None, label=None) -> System:
    c = cr.make(crystal)
    return System(label or f"Heisenberg {crystal}", 3, c, (L,) * c.D, (True,) * c.D, {1: J}, T_ref=T_ref)


def nanodot(Rd: float = 18.30, L: int = 5, Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
            KDM=0.88, gamma_deg=90.0, shells=(1,), J_shells=None) -> System:
    """The nanodot system: sc Heisenberg disc inside an open box.

    `shells` adds sc exchange shells beyond the first, with couplings `J_shells` {shell: J} relative
    to Jex; the colouring search then finds a colouring admissible for all of them. DMI stays on the
    first-shell in-plane bonds and the surface is still defined by first neighbours.

    The disc never reaches the box edge, so every seam bond would touch a masked (zero) site; an
    open box is the same Hamiltonian as a periodic one and meets condition (8) without a constraint on N.

    Bulk-type (Bloch) DMI restricted to the in-plane bonds: its vector lies along each bond, D_ij = -KDM r_ij,
    so the bond energy is -KDM r_ij . (S_i x S_j) (local field term KDM * (-dyz, dxz, dyx - dxy)), cubic anisotropy with surface constant KanS where a site has
    fewer than six in-disc neighbours, and a Zeeman field Hex (cos g, 0, sin g).
    """
    shells = tuple(sorted(set(shells) | {1}))
    c = cr.make("sc", shells)
    J = {1: Jex, **{int(k): float(v) for k, v in (J_shells or {}).items() if int(k) in shells and int(k) != 1}}
    L2 = int(Rd) + 1
    N = 2 * L2 + 1
    xs = np.arange(-L2, L2 + 1)
    xg, yg = np.meshgrid(xs, xs, indexing="xy")
    disk = (xg ** 2 + yg ** 2) <= Rd ** 2               # [y, x]
    mask = np.broadcast_to(disk, (L, N, N)).copy()[None]  # (1, L, N, N)
    surface = np.zeros_like(mask)
    for z in range(L):
        for y in range(N):
            for x in range(N):
                if not mask[0, z, y, x]:
                    continue
                k = 0
                for dz, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)):
                    nz, ny, nx = z + dz, (y + dy) % N, (x + dx) % N
                    if 0 <= nz < L and mask[0, nz, ny, nx]:
                        k += 1
                surface[0, z, y, x] = k < 6
    k1 = np.where(surface, KanS, Kan1).astype(np.float32) * mask
    extra = {}
    for i, bd in enumerate(c.bonds):
        dx, dy, dz = bd.delta
        E = np.zeros((3, 3))
        if (dx, dy, dz) == (1, 0, 0):
            E[1, 2], E[2, 1] = KDM, -KDM
        elif (dx, dy, dz) == (-1, 0, 0):
            E[1, 2], E[2, 1] = -KDM, KDM
        elif (dx, dy, dz) == (0, 1, 0):
            E[0, 2], E[2, 0] = -KDM, KDM
        elif (dx, dy, dz) == (0, -1, 0):
            E[0, 2], E[2, 0] = KDM, -KDM
        if E.any():
            extra[i] = E
    g = np.deg2rad(gamma_deg)
    return System("Heisenberg nanodot", 3, c, (N, N, L), (False, False, False), J, extra,
                  h=(Hex * float(np.cos(g)), 0.0, Hex * float(np.sin(g))), cubic_k1=k1, cubic_k2=Kan2,
                  mask=mask)


# Reference critical temperatures (J = kB = 1), from experiments v03 section E4.
TC = {
    "Ising square": 2.0 / np.log(1 + np.sqrt(2)),
    "Ising triangular": 4.0 / np.log(3.0),
    "Ising sc": 4.511523,      # 1/0.22165463, Ferrenberg, Xu and Landau 2018
    "Ising bcc": 6.355,        # 1/0.157373, Lundow and Campbell 2018
    "Ising fcc": 9.797132,     # 1/0.1020707(2), Yu 2015 (Physica A 419, 75)
    "XY sc": 2.20184,          # 1/0.4541652, Xu et al. 2019
    "Heisenberg sc": 1.4430,   # Chen, Ferrenberg and Landau 1993
    "Heisenberg bcc": 2.0542,  # Chen, Ferrenberg and Landau 1993
}


def matrix(small: bool = True) -> list[System]:
    """The 13 systems of experiment E1 at a small size per crystal (seam-compatible boxes)."""
    s = [
        ising("square", 8, T_ref=TC["Ising square"]),
        ising("triangular", 6, T_ref=TC["Ising triangular"]),
        ising("sc", 4, T_ref=TC["Ising sc"]),
        ising("bcc", 4, T_ref=TC["Ising bcc"]),
        ising("fcc", 4, T_ref=TC["Ising fcc"]),
        xy("square", 8, T_ref=0.893),
        xy("sc", 4, T_ref=TC["XY sc"]),
        heisenberg("sc", 4, T_ref=TC["Heisenberg sc"]),
        heisenberg("bcc", 4, T_ref=TC["Heisenberg bcc"]),
        heisenberg("fcc", 4, T_ref=None),
        heisenberg("triangular", 6, J=-1.0, T_ref=0.285, label="Heisenberg triangular AFM"),
        heisenberg("kagome", 6, J=-1.0, T_ref=None, label="Heisenberg kagome AFM"),
        nanodot(Rd=3.0, L=3),
    ]
    return s
