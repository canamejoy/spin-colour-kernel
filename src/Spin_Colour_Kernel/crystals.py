"""Crystals as Bravais lattices with a basis, and their bond tables (proposal r07, eq. 4).

A site is the pair (c, alpha): an integer cell coordinate c in Z^D and a basis index alpha.
A directed bond (alpha, alpha2, delta) joins (c, alpha) to (c + delta, alpha2). Bond tables are
generated from neighbour shells by real-space distance, so a crystal is fully described by its
primitive (or conventional) vectors and its basis offsets.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Bond:
    """One directed bond-table entry: (c, alpha) ~ (c + delta, alpha2), on neighbour shell `shell`."""

    alpha: int
    alpha2: int
    delta: tuple[int, ...]
    shell: int


@dataclass
class Crystal:
    name: str
    vectors: np.ndarray          # (D, D) rows are lattice vectors a_k in Cartesian units
    basis: np.ndarray            # (b, D) basis offsets in fractional (cell) coordinates
    shells: tuple[int, ...] = (1,)
    bonds: list[Bond] = field(default_factory=list)

    @property
    def D(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def b(self) -> int:
        return int(self.basis.shape[0])

    def position(self, c, alpha) -> np.ndarray:
        frac = np.asarray(c, dtype=float) + self.basis[alpha]
        return frac @ self.vectors


def shell_distances(vectors: np.ndarray, basis: np.ndarray, nshell: int, reach: int = 3) -> list[float]:
    """The first `nshell` distinct neighbour distances over every basis pair."""
    D = vectors.shape[0]
    ds = set()
    for a in range(basis.shape[0]):
        for a2 in range(basis.shape[0]):
            for delta in itertools.product(range(-reach, reach + 1), repeat=D):
                r = (np.asarray(delta, float) + basis[a2] - basis[a]) @ vectors
                d = float(np.linalg.norm(r))
                if d > 1e-9:
                    ds.add(round(d, 9))
    return sorted(ds)[:nshell]


def build_bonds(vectors: np.ndarray, basis: np.ndarray, shells: tuple[int, ...], reach: int = 3) -> list[Bond]:
    """Directed bond table for the requested shells (1-based), in a fixed, documented order.

    Order: shell, then alpha, then alpha2, then delta in lexicographic order. The order is the
    accumulation order of the local field, both in the kernel and in every reference.
    """
    dist = shell_distances(vectors, basis, max(shells), reach)
    D = vectors.shape[0]
    out: list[Bond] = []
    for s in sorted(shells):
        target = dist[s - 1]
        for a in range(basis.shape[0]):
            for a2 in range(basis.shape[0]):
                for delta in itertools.product(range(-reach, reach + 1), repeat=D):
                    r = (np.asarray(delta, float) + basis[a2] - basis[a]) @ vectors
                    if abs(np.linalg.norm(r) - target) < 1e-7:
                        out.append(Bond(a, a2, tuple(int(v) for v in delta), s))
    return out


def make(name: str, shells: tuple[int, ...] = (1,)) -> Crystal:
    """The crystals of the system matrix. bcc and fcc use conventional cubic cells (b = 2, 4)."""
    s3 = np.sqrt(3.0)
    tri = np.array([[1.0, 0.0], [0.5, s3 / 2]])
    table = {
        "square": (np.eye(2), np.zeros((1, 2))),
        "triangular": (tri, np.zeros((1, 2))),
        "honeycomb": (tri, np.array([[0.0, 0.0], [1 / 3, 1 / 3]])),
        "kagome": (tri, np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])),
        "sc": (np.eye(3), np.zeros((1, 3))),
        "bcc": (np.eye(3), np.array([[0, 0, 0], [0.5, 0.5, 0.5]])),
        "fcc": (np.eye(3), np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])),
    }
    if name not in table:
        raise KeyError(f"unknown crystal {name!r}; known: {sorted(table)}")
    vec, basis = table[name]
    vec = np.asarray(vec, float)
    basis = np.asarray(basis, float)
    reach = 3 if max(shells) <= 2 else 4
    return Crystal(name, vec, basis, tuple(shells), build_bonds(vec, basis, tuple(shells), reach))


def coordination(crystal: Crystal) -> dict[int, int]:
    """Bonds leaving each basis site, per alpha."""
    out = {a: 0 for a in range(crystal.b)}
    for bd in crystal.bonds:
        out[bd.alpha] += 1
    return out
