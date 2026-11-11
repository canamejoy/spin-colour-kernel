"""Linear colourings as lattice homomorphisms into a finite abelian group (proposal r07, eqs. 5-8).

gamma(c, alpha) = K c + o_alpha (mod m), with K in Z^{r x D} and o_alpha in Z_m1 x ... x Z_mr.
Admissibility (7) is a finite check over the bond table; the seam condition (8) binds the colouring
to a periodic box. The search runs over every abelian group (invariant-factor form) in increasing
order n_c and returns the first admissible colouring. The clique number of the bond graph is the
lower bound, so optimality is certified when the two meet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .crystals import Crystal


@dataclass(frozen=True)
class Colouring:
    m: tuple[int, ...]                 # group orders, invariant-factor form
    K: tuple[tuple[int, ...], ...]     # r rows, D columns
    o: tuple[tuple[int, ...], ...]     # b rows, r columns

    @property
    def n_colours(self) -> int:
        return int(np.prod(self.m)) if self.m else 1

    def colour(self, c, alpha) -> int:
        """Mixed-radix index of gamma(c, alpha) in [0, n_c)."""
        idx = 0
        for r, mr in enumerate(self.m):
            g = (sum(self.K[r][k] * int(c[k]) for k in range(len(c))) + self.o[alpha][r]) % mr
            idx = idx * mr + g
        return idx

    def describe(self) -> str:
        grp = " x ".join(f"Z{m}" for m in self.m) or "trivial"
        return f"{grp}; K={[list(r) for r in self.K]}; o={[list(x) for x in self.o]}"


def abelian_groups(n: int) -> list[tuple[int, ...]]:
    """Every abelian group of order n, as invariant factors m1 | m2 | ... | mr."""
    out = []

    def rec(rem, prev, acc):
        if rem == 1:
            out.append(tuple(acc))
            return
        for d in range(2, rem + 1):
            if rem % d == 0 and (prev is None or d % prev == 0):
                rec(rem // d, d, acc + [d])

    if n == 1:
        return [()]
    rec(n, None, [])
    # invariant factors must divide each other in order; rec builds m1 | m2 | ...
    return [g for g in out if all(g[i + 1] % g[i] == 0 for i in range(len(g) - 1))]


def admissible(col: Colouring, crystal: Crystal) -> bool:
    """Condition (7): K delta + o_alpha2 - o_alpha != 0 (mod m) for every bond."""
    for bd in crystal.bonds:
        v = []
        for r, mr in enumerate(col.m):
            v.append((sum(col.K[r][k] * bd.delta[k] for k in range(crystal.D))
                      + col.o[bd.alpha2][r] - col.o[bd.alpha][r]) % mr)
        if all(x == 0 for x in v):
            return False
    return True


def seam_ok(col: Colouring, box: tuple[int, ...], periodic: tuple[bool, ...]) -> bool:
    """Condition (8): every periodic axis k has K[:, k] L_k = 0 (mod m)."""
    for k, (Lk, per) in enumerate(zip(box, periodic)):
        if per and any((col.K[r][k] * Lk) % mr for r, mr in enumerate(col.m)):
            return False
    return True


def search(crystal: Crystal, box: tuple[int, ...] | None = None, periodic: tuple[bool, ...] | None = None,
           max_colours: int = 32) -> Colouring | None:
    """First admissible linear colouring in increasing n_c (and seam-compatible with `box` if given)."""
    D, b = crystal.D, crystal.b
    for n in range(1, max_colours + 1):
        for m in abelian_groups(n):
            r = len(m)
            elems = list(itertools.product(*[range(mi) for mi in m])) if r else [()]
            # K columns: one group element per lattice axis
            for cols in itertools.product(elems, repeat=D):
                K = tuple(tuple(cols[k][i] for k in range(D)) for i in range(r))
                for offs in itertools.product(elems, repeat=b - 1):
                    o = (tuple([0] * r),) + tuple(offs)
                    col = Colouring(m, K, o)
                    if box is not None and not seam_ok(col, box, periodic or (True,) * D):
                        continue
                    if admissible(col, crystal):
                        return col
    return None


def verify_on_box(col: Colouring, crystal: Crystal, box: tuple[int, ...], periodic: tuple[bool, ...]) -> bool:
    """Independent check: colour every site of the box and test every bond explicitly, seam included."""
    for c in itertools.product(*[range(L) for L in box]):
        for bd in crystal.bonds:
            if bd.alpha >= crystal.b:
                continue
            c2 = []
            inside = True
            for k in range(crystal.D):
                x = c[k] + bd.delta[k]
                if periodic[k]:
                    x %= box[k]
                elif not 0 <= x < box[k]:
                    inside = False
                c2.append(x)
            if inside and col.colour(c, bd.alpha) == col.colour(c2, bd.alpha2):
                return False
    return True


def clique_number(crystal: Crystal, window: int = 2) -> int:
    """Largest clique of the infinite bond graph, by Bron-Kerbosch on a window around one site.

    Every clique is translated so that one of its sites is (0, alpha); every other site lies within
    one bond of it, so a window of the bond reach holds every clique.
    """
    D = crystal.D
    reach = max(max(abs(x) for x in bd.delta) for bd in crystal.bonds)
    w = max(window, reach)
    nodes = [(c, a) for c in itertools.product(range(-w, w + 1), repeat=D) for a in range(crystal.b)]
    index = {n: i for i, n in enumerate(nodes)}

    nbrs = [set() for _ in nodes]
    for i, u in enumerate(nodes):
        for bd in crystal.bonds:
            if bd.alpha != u[1]:
                continue
            c2 = tuple(u[0][k] + bd.delta[k] for k in range(D))
            j = index.get((c2, bd.alpha2))
            if j is not None:
                nbrs[i].add(j)
    best = 1
    for a in range(crystal.b):
        start = index[(tuple([0] * D), a)]

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                best = max(best, len(R))
                return
            if len(R) + len(P) <= best:
                return
            for v in list(P):
                bk(R | {v}, P & nbrs[v], X & nbrs[v])
                P = P - {v}
                X = X | {v}

        bk({start}, set(nbrs[start]), set())
    return best
