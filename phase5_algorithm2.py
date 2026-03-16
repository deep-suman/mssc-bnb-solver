"""
Phase 5c: Algorithm 2 — Auxiliary Problem Solver (General Euclidean Space)
===========================================================================
Reference: Section 4.2, Algorithm 2 of the paper.

Algorithm 2 (paper, p.209-210):
  While G is not empty:
    (a) Find vertex n_i with smallest degree in G
    (b) Form subgraph G_i of n_i and its neighbors
    (c) Solve eq. (11) for variables in G_i  [via Dinkelbach]
    (d) Save clique if it is the best found so far
    (e) Remove n_i and its adjacent edges from G

Return the best cluster found.

Why this works (Proposition 4):
  The optimal cluster must be a clique in G.
  By iterating over all vertices and their neighborhoods, we implicitly
  enumerate all maximal cliques while pruning via the graph structure.

The sparsity of G depends on lambda values — when k is large,
lambda values are small → sparse G → Algorithm 2 is very fast.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase5_graph import IntersectionGraph, build_intersection_graph
from phase5_dinkelbach import dinkelbach


# ---------------------------------------------------------------------------
# 1.  Result structure
# ---------------------------------------------------------------------------

@dataclass
class AuxResult2:
    """
    Result of the general Euclidean auxiliary problem.

    Attributes
    ----------
    reduced_cost  : float         — best reduced cost found
    cluster       : Cluster|None  — new cluster to add (None if rc >= 0)
    n_cliques     : int           — number of cliques evaluated
    """
    reduced_cost : float
    cluster      : Optional[Cluster]
    n_cliques    : int

    def has_negative_rc(self, tol: float = 1e-6) -> bool:
        return self.reduced_cost < -tol


# ---------------------------------------------------------------------------
# 2.  Algorithm 2
# ---------------------------------------------------------------------------

def solve_auxiliary_general(inst: MSSCInstance,
                             lam: np.ndarray,
                             sigma: float) -> AuxResult2:
    """
    Solve the auxiliary problem for general Euclidean space.

    Parameters
    ----------
    inst  : MSSCInstance  (any dimension s)
    lam   : np.ndarray shape (n,)
    sigma : float

    Returns
    -------
    AuxResult2
    """
    # --- Build intersection graph ---
    G = build_intersection_graph(inst, lam)

    # Working copy of adjacency (we remove nodes during the loop)
    # We track active nodes and their current neighbors
    active      = list(range(inst.n))
    active_set  = set(active)
    adj         = [set(G.neighbors(i)) for i in range(inst.n)]

    best_rc    = -1e-6     # only accept genuinely negative reduced costs
    best_idx   = None
    n_cliques  = 0

    # --- Algorithm 2 main loop ---
    while active:
        # (a) Find active vertex with smallest degree
        ni = min(active, key=lambda i: len(adj[i] & active_set))

        # (b) Form G_i: ni and its active neighbors
        neighbors_i = sorted(adj[ni] & active_set)
        clique_nodes = [ni] + neighbors_i   # ni + all its current neighbors

        # (c) Solve the fractional QP on this subgraph via Dinkelbach
        rc, idx = dinkelbach(inst, lam, sigma, clique_nodes)
        n_cliques += 1

        # (d) Update best if improved
        if rc < best_rc and idx is not None and len(idx) > 0:
            best_rc  = rc
            best_idx = idx

        # (e) Remove ni and its edges from the active graph
        active_set.discard(ni)
        active.remove(ni)
        for nb in neighbors_i:
            adj[nb].discard(ni)

    # --- Build cluster ---
    best_cluster = None
    if best_idx is not None:
        best_cluster = make_cluster(inst, best_idx)

    return AuxResult2(
        reduced_cost = best_rc,
        cluster      = best_cluster,
        n_cliques    = n_cliques
    )


# ---------------------------------------------------------------------------
# 3.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance
    from phase2_jmeans import jmeans, extract_initial_columns
    from phase3_rmp import RestrictedMasterProblem

    print("=" * 50)
    print("Phase 5c — Algorithm 2 Tests")
    print("=" * 50)

    # --- Test 1: basic call on 2D instance ---
    inst1 = make_synthetic_instance(n=20, k=3, s=2, seed=0)
    lam1  = np.ones(inst1.n) * 5.0
    res1  = solve_auxiliary_general(inst1, lam1, sigma=1.0)
    assert isinstance(res1.reduced_cost, float)
    assert res1.n_cliques > 0
    print(f"✓ 2D result: rc={res1.reduced_cost:.4f}, "
          f"cliques={res1.n_cliques}")

    # --- Test 2: works on higher dimensional instance ---
    inst2 = make_synthetic_instance(n=20, k=3, s=5, seed=1)
    lam2  = np.ones(inst2.n) * 10.0
    res2  = solve_auxiliary_general(inst2, lam2, sigma=1.0)
    print(f"✓ 5D result: rc={res2.reduced_cost:.4f}, "
          f"cliques={res2.n_cliques}")

    # --- Test 3: sparse graph (small lambda) → fewer cliques ---
    inst3  = make_synthetic_instance(n=30, k=5, s=3, seed=2)
    lam_big   = np.full(inst3.n, 100.0)
    lam_small = np.full(inst3.n, 0.01)
    res_big   = solve_auxiliary_general(inst3, lam_big,   sigma=0.0)
    res_small = solve_auxiliary_general(inst3, lam_small, sigma=0.0)
    print(f"✓ Dense G  (big λ):   cliques={res_big.n_cliques}")
    print(f"✓ Sparse G (small λ): cliques={res_small.n_cliques}")

    # --- Test 4: full column generation loop on 4D instance ---
    inst4  = make_synthetic_instance(n=20, k=4, s=4, spread=1.0, seed=3)
    jm_sol = jmeans(inst4, n_restarts=3, seed=3)
    cols   = extract_initial_columns(inst4, jm_sol)

    rmp = RestrictedMasterProblem(inst4)
    rmp.add_columns(cols)

    print(f"\n✓ Column generation loop (n={inst4.n}, k={inst4.k}, s={inst4.s})")
    print(f"  Initial UB = {jm_sol.cost:.4f}")

    prev_obj = np.inf
    for it in range(30):
        obj, lam, sigma = rmp.solve()
        res = solve_auxiliary_general(inst4, lam, sigma)

        print(f"  Iter {it+1:2d}: LP={obj:.4f}  "
              f"rc={res.reduced_cost:.6f}  "
              f"cliques={res.n_cliques}")

        if not res.has_negative_rc():
            print(f"  ✓ Optimality certified at iter {it+1}")
            break

        if obj >= prev_obj - 1e-8:
            print(f"  ✓ LP not improving → optimal: LP={obj:.4f}")
            break

        prev_obj = obj
        rmp.add_column(res.cluster)
    else:
        print(f"  ! Did not converge in 30 iters")

    print("\nAll Algorithm 2 tests passed.")
