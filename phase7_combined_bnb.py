"""
Phase 7: Combined B&B Solver — Multi-Column Pricing + Pruned Auxiliary
=======================================================================
Improves on phase6_bnb_optimized.py by using multi-column pricing at the
root node: instead of adding one column per CG iteration, ALL improving
columns found in one graph traversal are added to the RMP before the next
LP solve.

What changes relative to phase6_bnb_optimized.py
--------------------------------------------------
  Root node (general, s != 2):
    solve_auxiliary_combined()  (phase7_pruned_multi_aux)
    → returns ALL clusters with rc < -1e-6 in one pass
    → added in bulk before next rmp.solve()
    → fewer LP (Gurobi barrier) solves to reach LP optimality

  Root node (s == 2):
    solve_auxiliary_2d() — already returns one column per call;
    Algorithm 1 geometry does not lend itself to multi-pricing, kept as-is.

  Non-root nodes (with branching constraints):
    _constrained_auxiliary_pruned() — identical to phase6_bnb_optimized;
    constraint-aware single-column fallback for correctness.

What is NOT changed
-------------------
  - B&B structure, Ryan-Foster branching, pruning: identical
  - solve_bnb_combined() returns BnBResult: same type, same fields
  - j-means initialisation, column seeding: identical
  - Gurobi RMP: original RestrictedMasterProblem

Original files are NOT modified.
"""

import numpy as np
import time
import heapq
from typing import Optional, List, Tuple

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase5_algorithm2_pruned import solve_auxiliary_pruned
from phase6_branching import (BranchNode, find_branching_pair,
                               is_integer_solution,
                               column_satisfies_constraints)
from phase6_constrained_aux import constrained_auxiliary_general
from phase6_bnb import BnBResult, _extract_labels
from phase7_pruned_multi_aux import solve_auxiliary_combined


# ---------------------------------------------------------------------------
# Constrained auxiliary (pruned, with fallback) — same as phase6_bnb_optimized
# ---------------------------------------------------------------------------

def _constrained_auxiliary_pruned(inst  : MSSCInstance,
                                   lam   : np.ndarray,
                                   sigma : float,
                                   node  : BranchNode):
    """
    Try pruned unconstrained auxiliary; fall back to constrained if the
    returned cluster violates branching constraints.
    """
    aux = solve_auxiliary_pruned(inst, lam, sigma)
    if (aux.cluster is not None and
            column_satisfies_constraints(aux.cluster, node)):
        return aux
    return constrained_auxiliary_general(inst, lam, sigma, node)


# ---------------------------------------------------------------------------
# Node solver with multi-column pricing at root
# ---------------------------------------------------------------------------

def _solve_node_combined(inst        : MSSCInstance,
                         node        : BranchNode,
                         all_columns : List[Cluster],
                         ub          : float,
                         max_cg_iter : int  = 200,
                         verbose     : bool = False
                         ) -> Tuple[float, np.ndarray, List[Cluster]]:
    """
    Solve LP relaxation at a B&B node.

    Root node (general s): multi-column pricing via solve_auxiliary_combined.
    Root node (s == 2)   : Algorithm 1 (single column per iter, unchanged).
    Non-root nodes       : constrained pruned auxiliary (single column).
    """
    valid_cols = [cl for cl in all_columns
                  if column_satisfies_constraints(cl, node)]

    rmp = RestrictedMasterProblem(inst)
    if not valid_cols:
        return np.inf, np.array([]), []
    rmp.add_columns(valid_cols)

    new_cols      = []
    prev_obj      = np.inf
    no_improve_ct = 0

    for _ in range(max_cg_iter):
        lp_obj, lam, sigma = rmp.solve()

        if inst.s == 2 and node.is_root():
            # Algorithm 1 for 2D root — single column, no change needed
            aux = solve_auxiliary_2d(inst, lam, sigma)
            rc  = aux.reduced_cost
            if rc >= -1e-6:
                break
            cl = aux.cluster
            if cl is not None and cl.indices not in rmp._col_index_sets:
                rmp.add_column(cl)
                new_cols.append(cl)

        elif node.is_root():
            # Multi-column pricing for general (s > 2) root node
            aux = solve_auxiliary_combined(inst, lam, sigma)
            if not aux.has_negative_rc():
                break
            for cl in aux.clusters:
                if cl.indices not in rmp._col_index_sets:
                    rmp.add_column(cl)
                    new_cols.append(cl)

        else:
            # Non-root: branching constraints require constrained auxiliary
            aux = _constrained_auxiliary_pruned(inst, lam, sigma, node)
            rc  = aux.reduced_cost
            if rc >= -1e-6:
                break
            cl = aux.cluster
            if (cl is not None and
                    column_satisfies_constraints(cl, node) and
                    cl.indices not in rmp._col_index_sets):
                rmp.add_column(cl)
                new_cols.append(cl)

        if lp_obj >= prev_obj - 1e-8:
            no_improve_ct += 1
            if no_improve_ct >= 10:
                break
        else:
            no_improve_ct = 0

        prev_obj = lp_obj

    lp_obj, _, _ = rmp.solve()
    z_values     = rmp.get_solution()

    return lp_obj, z_values if z_values is not None else np.array([]), new_cols


# ---------------------------------------------------------------------------
# Combined B&B main loop
# ---------------------------------------------------------------------------

def solve_bnb_combined(inst             : MSSCInstance,
                       max_nodes        : int  = 200,
                       n_jmeans_restarts: int  = 5,
                       seed             : int  = 42,
                       verbose          : bool = True) -> BnBResult:
    """
    Exact MSSC solver: B&B + multi-column pricing + pruned auxiliary.

    Same interface and return type as solve_bnb() in phase6_bnb.py.

    Parameters
    ----------
    inst              : MSSCInstance
    max_nodes         : B&B node limit
    n_jmeans_restarts : j-means restarts for initial upper bound
    seed              : RNG seed
    verbose           : print B&B log
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Combined B&B  |  {inst.name}  "
              f"|  n={inst.n}, k={inst.k}, s={inst.s}")
        print(f"{'='*60}")

    jm_sol      = jmeans(inst, n_restarts=n_jmeans_restarts, seed=seed)
    ub          = jm_sol.cost
    best_labels = jm_sol.labels
    all_columns = extract_initial_columns(inst, jm_sol)
    all_col_indices: set = {cl.indices for cl in all_columns}

    if verbose:
        print(f"  Initial UB = {ub:.6f}")

    root    = BranchNode()
    node_id = 0
    heap    = [(0.0, node_id, root)]
    n_nodes = 0
    lp_bound = 0.0

    while heap and n_nodes < max_nodes:
        lb_est, _, node = heapq.heappop(heap)

        if lb_est >= ub - 1e-6:
            if verbose:
                print(f"  Node {n_nodes:3d}: PRUNED "
                      f"(lb={lb_est:.4f} >= UB={ub:.4f})")
            continue

        n_nodes += 1

        lp_obj, z_values, new_cols = _solve_node_combined(
            inst, node, all_columns, ub, verbose=False)

        for cl in new_cols:
            if cl.indices not in all_col_indices:
                all_columns.append(cl)
                all_col_indices.add(cl.indices)

        if node.is_root():
            lp_bound = lp_obj

        if verbose:
            print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                  f"LP={lp_obj:.4f}  UB={ub:.4f}  "
                  f"new_cols={len(new_cols)}")

        if lp_obj >= ub - 1e-6:
            if verbose:
                print(f"         -> PRUNED")
            continue

        if len(z_values) > 0 and is_integer_solution(z_values):
            valid_cols = [cl for cl in all_columns
                          if column_satisfies_constraints(cl, node)]
            labels = _extract_labels(inst, valid_cols, z_values)
            if labels is not None and lp_obj < ub - 1e-6:
                ub          = lp_obj
                best_labels = labels
                if verbose:
                    print(f"         -> NEW UB = {ub:.6f}")
            continue

        if len(z_values) == 0:
            continue

        valid_cols = [cl for cl in all_columns
                      if column_satisfies_constraints(cl, node)]
        pair = find_branching_pair(valid_cols, z_values)
        if pair is None:
            continue

        i1, i2 = pair
        if verbose:
            print(f"         -> BRANCH on ({i1},{i2})")

        for child in [node.branch_same(i1, i2), node.branch_diff(i1, i2)]:
            node_id += 1
            heapq.heappush(heap, (lp_obj, node_id, child))

    elapsed = time.time() - t_start
    gap_pct = ((ub - lp_bound) / lp_bound * 100.0
               if lp_bound > 1e-10 else 0.0)
    status  = "optimal" if n_nodes < max_nodes else "max_nodes"

    if verbose:
        print(f"\n  {'─'*58}")
        print(f"  Status       : {status}")
        print(f"  Optimal cost : {ub:.6f}")
        print(f"  LP bound     : {lp_bound:.6f}")
        print(f"  Gap          : {gap_pct:.4f}%")
        print(f"  B&B nodes    : {n_nodes}")
        print(f"  Columns      : {len(all_columns)}")
        print(f"  Time         : {elapsed:.1f}s")

    return BnBResult(
        optimal_cost   = ub,
        optimal_labels = best_labels,
        lp_bound       = lp_bound,
        gap_pct        = gap_pct,
        n_nodes        = n_nodes,
        n_columns      = len(all_columns),
        time_sec       = elapsed,
        status         = status,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance, MSSCInstance
    import numpy as np

    print("=" * 60)
    print("Phase 7 — Combined B&B Tests")
    print("=" * 60)

    # Test 1: well-separated 2D (should need 0 branches, solved at root)
    inst1 = make_synthetic_instance(n=30, k=3, s=2, spread=0.3, seed=0)
    res1  = solve_bnb_combined(inst1, verbose=True)
    assert res1.gap_pct < 1e-3, f"Expected 0% gap, got {res1.gap_pct:.4f}%"
    print(f"\n✓ Test 1: {res1}\n")

    # Test 2: known optimal synthetic instance
    centers = np.array([[0., 0.], [50., 0.], [25., 43.]])
    pts     = np.vstack([
        centers[0] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
        centers[1] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
        centers[2] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
    ])
    inst2    = MSSCInstance(points=pts, k=3, name="known_opt")
    true_opt = sum(
        ((pts[c*5:(c+1)*5] - pts[c*5:(c+1)*5].mean(axis=0))**2).sum()
        for c in range(3)
    )
    res2 = solve_bnb_combined(inst2, verbose=False)
    assert abs(res2.optimal_cost - true_opt) < 1e-3, \
        f"Expected {true_opt:.4f}, got {res2.optimal_cost:.4f}"
    print(f"✓ Test 2: known optimal {true_opt:.4f}, found {res2.optimal_cost:.4f}")

    # Test 3: higher-dimensional (multi-column pricing active)
    inst3 = make_synthetic_instance(n=20, k=3, s=4, spread=1.0, seed=2)
    res3  = solve_bnb_combined(inst3, verbose=False)
    assert res3.optimal_cost > 0
    print(f"✓ Test 3 (s=4): {res3}")

    print("\nAll Phase 7 B&B tests passed.")
