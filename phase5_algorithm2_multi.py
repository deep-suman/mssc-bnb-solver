"""
Phase 5c (multi): Auxiliary Problem — Multi-Column Pricing.

Returns ALL columns with negative reduced cost found during one pass of
Algorithm 2, rather than just the best one. This reduces the number of
LP re-solves per B&B node by adding multiple improving columns at once.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase5_graph_vectorized import build_intersection_graph_vectorized
from phase5_dinkelbach import dinkelbach


@dataclass
class AuxResultMulti:
    best_reduced_cost: float
    columns: List[Cluster] = field(default_factory=list)
    n_cliques: int = 0

    def has_negative_rc(self, tol: float = 1e-6) -> bool:
        return self.best_reduced_cost < -tol


def solve_auxiliary_multi(inst: MSSCInstance,
                          lam: np.ndarray,
                          sigma: float,
                          max_columns: int = 10) -> AuxResultMulti:
    """
    Run Algorithm 2 and return up to max_columns negative-RC columns.

    Parameters
    ----------
    inst        : MSSCInstance
    lam         : np.ndarray shape (n,)
    sigma       : float
    max_columns : int   — max columns to return per call

    Returns
    -------
    AuxResultMulti
    """
    n = inst.n
    best_rc = -1e-6
    columns: List[Cluster] = []
    seen_indices = set()
    n_cliques = 0

    # Singleton shortcuts
    for i in range(n):
        rc_i = sigma - lam[i]
        if rc_i < best_rc:
            col = make_cluster(inst, [i])
            key = frozenset([i])
            if key not in seen_indices:
                seen_indices.add(key)
                columns.append(col)
                best_rc = rc_i

    G = build_intersection_graph_vectorized(inst, lam)
    adj = [set(G.neighbors(i)) for i in range(n)]

    active = set(range(n))
    active_set = set(range(n))

    while active and len(columns) < max_columns:
        ni = min(active, key=lambda i: len(adj[i] & active_set))
        active.discard(ni)

        clique_nodes = [ni] + [j for j in adj[ni] if j in active_set]
        n_cliques += 1

        rc, idx = dinkelbach(inst, lam, sigma, clique_nodes)

        if rc < -1e-6 and idx is not None and len(idx) > 0:
            key = frozenset(idx)
            if key not in seen_indices:
                seen_indices.add(key)
                col = make_cluster(inst, list(idx))
                columns.append(col)
                if rc < best_rc:
                    best_rc = rc

        active_set.discard(ni)

    return AuxResultMulti(
        best_reduced_cost=best_rc,
        columns=columns,
        n_cliques=n_cliques,
    )
