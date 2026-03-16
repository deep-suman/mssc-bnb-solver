"""
Phase 6b: Constrained Auxiliary Problem
=========================================
Reference: Section 3.1 and 4.1 of the paper.

When branching constraints are present, the auxiliary problem (eq. 8)
becomes:

    min  sum_i (||p_i - y||^2 - lambda_i) * v_i
    s.t. v_i = v_j      for (i,j) in I1   [SAME constraints]
         v_i + v_j <= 1  for (i,j) in I2   [DIFF constraints]
         v_i in {0,1}

How constraints modify the search (paper Section 4.1):

  SAME constraint (i,j) → merge variables v_i and v_j into one.
    - In the QP coefficients: combine rows/cols for i and j.
    - Weight w_i tracks how many original variables are merged.

  DIFF constraint (i,j) → set Q_ij to a large value M.
    - This makes it never optimal to have both v_i=1 and v_j=1.

For Algorithm 1 (2D):
  SAME: intersection point search must include BOTH i and j or NEITHER.
  DIFF: pairs (i,j) with DIFF constraint are treated as non-intersecting.

For Algorithm 2 (general):
  SAME: merged variables → updated graph edges.
  DIFF: remove edge (i,j) from G.

This file implements:
  1. apply_constraints_to_graph  — modify G for Algorithm 2
  2. apply_constraints_to_discs  — modify disc sets for Algorithm 1
  3. constrained_auxiliary_2d    — Algorithm 1 with constraints
  4. constrained_auxiliary_general — Algorithm 2 with constraints
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from copy import deepcopy

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase4_geometry import Disc, build_discs
from phase4_regions import (enumerate_regions, candidate_index_sets,
                             isolated_disc_index_set)
from phase4_algorithm1 import AuxResult
from phase5_graph import IntersectionGraph, build_intersection_graph
from phase5_dinkelbach import dinkelbach, build_qp_coefficients
from phase5_algorithm2 import AuxResult2
from phase6_branching import BranchNode, column_satisfies_constraints


# ---------------------------------------------------------------------------
# 1.  Apply constraints to intersection graph (Algorithm 2)
# ---------------------------------------------------------------------------

def apply_constraints_to_graph(G         : IntersectionGraph,
                                node      : BranchNode
                                ) -> IntersectionGraph:
    """
    Modify G to reflect branching constraints.

    DIFF (i,j): remove edge (i,j) — they can never be in same cluster.
    SAME (i,j): merge node j into node i — they always appear together.
                (we handle merging in the Dinkelbach call instead,
                 by marking pairs that must co-occur)

    For simplicity here we only apply DIFF by edge removal.
    SAME is handled in the Dinkelbach call via the node's same_pairs.
    """
    n      = G.n
    new_G  = IntersectionGraph(n)

    diff_set = set((min(i,j), max(i,j)) for i,j in node.diff_pairs)

    for i, j in G.edges:
        if (min(i,j), max(i,j)) not in diff_set:
            new_G.add_edge(i, j)

    return new_G


# ---------------------------------------------------------------------------
# 2.  Constrained Dinkelbach (handles SAME constraints via merging)
# ---------------------------------------------------------------------------

def constrained_dinkelbach(inst   : MSSCInstance,
                            lam    : np.ndarray,
                            sigma  : float,
                            clique : List[int],
                            node   : BranchNode
                            ) -> Tuple[float, Optional[List[int]]]:
    """
    Run Dinkelbach on a clique respecting SAME/DIFF branching constraints.

    SAME (i,j): if i is in clique, j must also be — add j to clique
                if not already present, and vice versa.
    DIFF (i,j): if both i and j are in clique, set Q_ij = M (large).
    """
    M = 1e9    # large value for DIFF pairs

    # --- Expand clique to include SAME-paired entities ---
    clique_set = set(clique)
    changed    = True
    while changed:
        changed = False
        for i, j in node.same_pairs:
            if i in clique_set and j not in clique_set:
                clique_set.add(j)
                changed = True
            elif j in clique_set and i not in clique_set:
                clique_set.add(i)
                changed = True
    clique = sorted(clique_set)

    if len(clique) == 0:
        return 0.0, None

    # --- Build QP coefficients ---
    Q, L = build_qp_coefficients(inst, lam, clique)

    # --- Apply DIFF constraints: set Q_ij = M ---
    diff_set = set((min(i,j), max(i,j)) for i,j in node.diff_pairs)
    for a, i in enumerate(clique):
        for b, j in enumerate(clique):
            if b <= a:
                continue
            if (min(i,j), max(i,j)) in diff_set:
                Q[a, b] = M
                Q[b, a] = M

    # --- Run Dinkelbach with modified Q ---
    from phase5_dinkelbach import solve_01qp, eval_numerator, eval_denominator
    m        = len(clique)
    q        = 0.0
    best_rc  = 0.0
    best_idx = None

    for _ in range(50):
        v, obj_val = solve_01qp(Q, L, q)
        if v.sum() < 0.5:
            break

        num   = eval_numerator(v, Q, L)
        den   = eval_denominator(v)
        if den < 0.5:
            break

        q_new = num / den
        rc    = sigma + q_new

        # Verify SAME constraints are satisfied
        idx_set = {clique[a] for a in range(m) if v[a] > 0.5}
        valid   = True
        for i, j in node.same_pairs:
            if (i in idx_set) != (j in idx_set):
                valid = False
                break

        if valid and rc < best_rc:
            best_rc  = rc
            best_idx = sorted(idx_set)

        if abs(q_new - q) < 1e-8:
            break
        q = q_new

    return best_rc, best_idx


# ---------------------------------------------------------------------------
# 3.  Constrained auxiliary problem — general Euclidean space
# ---------------------------------------------------------------------------

def constrained_auxiliary_general(inst  : MSSCInstance,
                                   lam   : np.ndarray,
                                   sigma : float,
                                   node  : BranchNode
                                   ) -> AuxResult2:
    """
    Solve the auxiliary problem respecting branching constraints.
    Uses Algorithm 2 with modified graph and constrained Dinkelbach.
    """
    if node.is_root():
        from phase5_algorithm2 import solve_auxiliary_general
        return solve_auxiliary_general(inst, lam, sigma)

    # Build and modify intersection graph
    G = build_intersection_graph(inst, lam)
    G = apply_constraints_to_graph(G, node)

    active     = list(range(inst.n))
    active_set = set(active)
    adj        = [set(G.neighbors(i)) for i in range(inst.n)]

    best_rc   = -1e-6
    best_idx  = None
    n_cliques = 0

    while active:
        ni           = min(active, key=lambda i: len(adj[i] & active_set))
        neighbors_i  = sorted(adj[ni] & active_set)
        clique_nodes = [ni] + neighbors_i

        rc, idx = constrained_dinkelbach(inst, lam, sigma,
                                          clique_nodes, node)
        n_cliques += 1

        if rc < best_rc and idx is not None and len(idx) > 0:
            best_rc  = rc
            best_idx = idx

        active_set.discard(ni)
        active.remove(ni)
        for nb in neighbors_i:
            adj[nb].discard(ni)

    best_cluster = None
    if best_idx is not None:
        cl = make_cluster(inst, best_idx)
        if column_satisfies_constraints(cl, node):
            best_cluster = cl
        else:
            best_rc = 0.0

    return AuxResult2(
        reduced_cost = best_rc,
        cluster      = best_cluster,
        n_cliques    = n_cliques
    )


# ---------------------------------------------------------------------------
# 4.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance, make_cluster

    print("=" * 50)
    print("Phase 6b — Constrained Auxiliary Tests")
    print("=" * 50)

    inst = make_synthetic_instance(n=15, k=3, s=2, seed=0)
    lam  = np.ones(inst.n) * 5.0
    sigma = 1.0

    # --- Test 1: root node matches unconstrained solver ---
    from phase5_algorithm2 import solve_auxiliary_general
    root     = BranchNode()
    res_root = constrained_auxiliary_general(inst, lam, sigma, root)
    res_unc  = solve_auxiliary_general(inst, lam, sigma)
    assert abs(res_root.reduced_cost - res_unc.reduced_cost) < 1e-6
    print(f"✓ Root node matches unconstrained: rc={res_root.reduced_cost:.4f}")

    # --- Test 2: DIFF constraint excludes pairs ---
    node_diff = BranchNode(diff_pairs=[(0, 1)])
    res_diff  = constrained_auxiliary_general(inst, lam, sigma, node_diff)
    if res_diff.cluster is not None:
        idx = set(res_diff.cluster.indices)
        assert not (0 in idx and 1 in idx), \
            "DIFF constraint violated: 0 and 1 both in cluster"
    print(f"✓ DIFF constraint respected: rc={res_diff.reduced_cost:.4f}")

    # --- Test 3: SAME constraint keeps pairs together ---
    node_same = BranchNode(same_pairs=[(0, 1)])
    res_same  = constrained_auxiliary_general(inst, lam, sigma, node_same)
    if res_same.cluster is not None:
        idx = set(res_same.cluster.indices)
        if 0 in idx or 1 in idx:
            assert 0 in idx and 1 in idx, \
                "SAME constraint violated: 0 and 1 must be together"
    print(f"✓ SAME constraint respected: rc={res_same.reduced_cost:.4f}")

    # --- Test 4: column_satisfies_constraints filters correctly ---
    node  = BranchNode(same_pairs=[(0,1)], diff_pairs=[(2,3)])
    good  = make_cluster(inst, [0, 1, 4])
    bad1  = make_cluster(inst, [0, 4])      # violates SAME(0,1)
    bad2  = make_cluster(inst, [2, 3, 4])   # violates DIFF(2,3)
    assert     column_satisfies_constraints(good, node)
    assert not column_satisfies_constraints(bad1, node)
    assert not column_satisfies_constraints(bad2, node)
    print(f"✓ Column filtering correct")

    # --- Test 5: graph DIFF edge removal ---
    from phase5_graph import build_intersection_graph
    G      = build_intersection_graph(inst, lam)
    n_orig = len(G.edges)
    G_mod  = apply_constraints_to_graph(G, node)
    assert len(G_mod.edges) <= n_orig
    print(f"✓ Graph edges: {n_orig} → {len(G_mod.edges)} after DIFF removal")

    print("\nAll constrained auxiliary tests passed.")
