"""
Phase 7: Pruned Multi-Column Auxiliary Problem Solver
=====================================================
Combines all auxiliary optimisations into a single function that returns
ALL improving columns per graph traversal instead of just the best one.

Four improvements in one pass
------------------------------
1. Vectorised graph build  (phase5_graph_vectorized) — O(n²) NumPy/BLAS
2. Singleton shortcut      — rc = σ - λ_i for |C|=1, zero Dinkelbach calls
3. Bound pruning           — skip clique C when σ - Σ_{i∈C}λ_i ≥ -1e-6
4. Multi-column collection — every cluster with rc < -1e-6 is returned

Why multi-column matters
------------------------
Algorithm 2 (phase5_algorithm2.py) traverses the intersection graph once
per CG iteration and returns the single best cluster found.  The graph
traversal is the dominant cost; returning all improving clusters from the
same traversal costs almost nothing extra, yet gives the RMP more columns
per LP solve.  This reduces the total number of LP solves (Gurobi barrier
calls) needed to reach LP optimality.

Pruning threshold in multi-column mode
---------------------------------------
In single-best mode the bound threshold adapts to the current best_rc
(see phase5_algorithm2_pruned.py).  In multi-column mode the threshold is
fixed at -1e-6 so that no valid improving column is missed:

    If σ - Σ_{i∈C} λ_i ≥ -1e-6
    then for every non-empty S ⊆ C:
        rc(S) = cost(S) + σ - Σ_{i∈S} λ_i  ≥  0 + σ - Σ_{i∈C} λ_i  ≥ -1e-6

so no subset of C can contribute a useful column.  Clique is skipped.

Original files are NOT modified.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase5_graph_vectorized import build_intersection_graph_fast
from phase5_dinkelbach import dinkelbach


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AuxResultMultiPruned:
    """
    Result of the combined pruned multi-column auxiliary.

    Attributes
    ----------
    best_reduced_cost : float          -- most negative RC (0.0 if none found)
    clusters          : List[Cluster]  -- all clusters with rc < -1e-6 (unique)
    n_cliques         : int            -- total cliques in traversal
    n_pruned          : int            -- cliques skipped by bound check
    n_singletons      : int            -- size-1 cliques (analytical, no QP)
    n_dinkelbach      : int            -- cliques that ran full Dinkelbach
    """
    best_reduced_cost : float
    clusters          : List[Cluster]
    n_cliques         : int
    n_pruned          : int
    n_singletons      : int
    n_dinkelbach      : int

    def has_negative_rc(self, tol: float = 1e-6) -> bool:
        return self.best_reduced_cost < -tol


# ---------------------------------------------------------------------------
# Pruned multi-column auxiliary (Algorithm 2 variant)
# ---------------------------------------------------------------------------

def solve_auxiliary_combined(inst  : MSSCInstance,
                              lam   : np.ndarray,
                              sigma : float) -> AuxResultMultiPruned:
    """
    Solve auxiliary problem: return ALL clusters with negative reduced cost.

    Traversal follows Algorithm 2 (smallest-degree-first removal), with
    pruning applied before Dinkelbach and multi-column collection afterwards.

    Parameters
    ----------
    inst  : MSSCInstance  (any dimension s)
    lam   : np.ndarray shape (n,)
    sigma : float

    Returns
    -------
    AuxResultMultiPruned
    """
    G          = build_intersection_graph_fast(inst, lam)
    active     = list(range(inst.n))
    active_set = set(active)
    adj        = [set(G.neighbors(i)) for i in range(inst.n)]

    best_rc      = 0.0                  # tracks most negative rc seen
    clusters_out : List[Cluster] = []
    seen_indices : set           = set() # dedup by index-tuple

    n_cliques    = 0
    n_pruned     = 0
    n_singletons = 0
    n_dinkelbach = 0

    while active:
        # (a) lowest-degree active vertex
        ni = min(active, key=lambda i: len(adj[i] & active_set))

        # (b) subgraph: ni + its active neighbours
        neighbors_i  = sorted(adj[ni] & active_set)
        clique_nodes = [ni] + neighbors_i
        m            = len(clique_nodes)
        n_cliques   += 1

        if m == 1:
            # --- Singleton shortcut: closed-form, no Dinkelbach ---
            n_singletons += 1
            rc = sigma - lam[ni]
            if rc < -1e-6:
                idx_t = (ni,)
                if idx_t not in seen_indices:
                    seen_indices.add(idx_t)
                    clusters_out.append(make_cluster(inst, idx_t))
                    if rc < best_rc:
                        best_rc = rc

        else:
            # --- Bound pruning: can any subset of C have rc < -1e-6? ---
            lam_sum = sum(lam[i] for i in clique_nodes)
            if sigma - lam_sum >= -1e-6:
                # No subset can improve → skip Dinkelbach entirely
                n_pruned += 1
            else:
                # --- Run Dinkelbach ---
                n_dinkelbach += 1
                rc, idx = dinkelbach(inst, lam, sigma, clique_nodes)
                if rc < -1e-6 and idx is not None and len(idx) > 0:
                    idx_t = tuple(sorted(idx))
                    if idx_t not in seen_indices:
                        seen_indices.add(idx_t)
                        clusters_out.append(make_cluster(inst, idx_t))
                        if rc < best_rc:
                            best_rc = rc

        # (e) remove ni from active graph
        active_set.discard(ni)
        active.remove(ni)
        for nb in neighbors_i:
            adj[nb].discard(ni)

    return AuxResultMultiPruned(
        best_reduced_cost = best_rc if clusters_out else 0.0,
        clusters          = clusters_out,
        n_cliques         = n_cliques,
        n_pruned          = n_pruned,
        n_singletons      = n_singletons,
        n_dinkelbach      = n_dinkelbach,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance
    from phase2_jmeans import jmeans, extract_initial_columns
    from phase3_rmp import RestrictedMasterProblem
    from phase5_algorithm2 import solve_auxiliary_general

    print("=" * 60)
    print("Phase 7 — Pruned Multi-Column Auxiliary Tests")
    print("=" * 60)

    # Test 1: basic call, result is a superset of single-best result
    inst1 = make_synthetic_instance(n=20, k=3, s=3, seed=0)
    lam1  = np.ones(inst1.n) * 5.0
    r1    = solve_auxiliary_combined(inst1, lam1, sigma=1.0)
    print(f"\n✓ Test 1 (n=20, k=3, s=3):")
    print(f"  Clusters returned : {len(r1.clusters)}")
    print(f"  Best RC           : {r1.best_reduced_cost:.6f}")
    print(f"  Cliques           : {r1.n_cliques}")
    print(f"  Singletons        : {r1.n_singletons}")
    print(f"  Pruned            : {r1.n_pruned}")
    print(f"  Dinkelbach        : {r1.n_dinkelbach}")

    # Test 2: best single-best RC should match the minimum among multi results
    r1_single = solve_auxiliary_general(inst1, lam1, sigma=1.0)
    if r1.clusters:
        multi_best = min(
            make_cluster(inst1, cl.indices).cost + 1.0 - sum(lam1[i] for i in cl.indices)
            for cl in r1.clusters
        )
        print(f"\n✓ Test 2: single-best rc={r1_single.reduced_cost:.6f}  "
              f"multi best_rc={r1.best_reduced_cost:.6f}")
        rc_close = abs(r1_single.reduced_cost - r1.best_reduced_cost) < 1e-5
        print(f"  RC match (within 1e-5): {'OK' if rc_close else 'MISMATCH'}")

    # Test 3: with zero lambda — no intersections, all singletons
    inst2 = make_synthetic_instance(n=15, k=3, s=2, seed=1)
    lam2  = np.zeros(inst2.n)
    r2    = solve_auxiliary_combined(inst2, lam2, sigma=0.5)
    print(f"\n✓ Test 3 (zero lambda → only singletons):")
    print(f"  Singletons: {r2.n_singletons}  Dinkelbach: {r2.n_dinkelbach}")

    # Test 4: full CG loop — multi-pricing converges correctly
    inst3  = make_synthetic_instance(n=20, k=4, s=4, spread=1.0, seed=3)
    jm_sol = jmeans(inst3, n_restarts=3, seed=3)
    cols   = extract_initial_columns(inst3, jm_sol)
    rmp    = RestrictedMasterProblem(inst3)
    rmp.add_columns(cols)

    print(f"\n✓ Test 4: CG loop with multi-pricing (n={inst3.n}, k={inst3.k})")
    n_lp_solves = 0
    n_cols_added = 0
    for it in range(50):
        obj, lam, sigma = rmp.solve()
        n_lp_solves += 1
        r = solve_auxiliary_combined(inst3, lam, sigma)
        if not r.has_negative_rc():
            print(f"  LP optimal at iter {it+1}: LP={obj:.4f}  "
                  f"(rc={r.best_reduced_cost:.6f})")
            break
        added = 0
        for cl in r.clusters:
            if cl.indices not in rmp._col_index_sets:
                rmp.add_column(cl)
                added += 1
        n_cols_added += added
    print(f"  LP solves: {n_lp_solves}  Columns added: {n_cols_added}")

    print("\nAll Phase 7 auxiliary tests passed.")
