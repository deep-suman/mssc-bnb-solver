"""
Phase 7b: Lean Warm-Start B&B
==============================
Speeds up child B&B nodes by starting their RMP with only the *basic*
columns (z > threshold) from the parent's LP solution, rather than the
full global column pool which grows to hundreds of columns.

Why this helps
--------------
After the root node's column generation, the global pool may contain
500+ columns. Every child node currently builds an RMP with all valid
columns from that pool, so Gurobi solves a 500-column barrier LP from
scratch at every child node.

At a fractional LP optimum, exactly k clusters are basic (z > 0) — one
per cluster in the LP solution.  Those k basic columns cover all n
entities (since covering constraints must be satisfied).  Child nodes
inherit an LP that differs from the parent only by branching constraints,
so the parent's basic columns are a correct and tight warm start.

Lean warm-start: child starts with (parent's basic columns) ∩ (valid for
child constraints).  Any entities left uncovered by constraint filtering
are patched from the global pool.  CG adds whatever else is needed.

Expected effect
---------------
Root node : unchanged — starts from j-means columns as before.
Child nodes: initial RMP is ~k columns instead of 500+.  Each Gurobi
  barrier solve is proportional to (n_cols)^p — fewer columns → faster
  LP solves.  CG re-adds needed columns; the global pool is still the
  source of truth.

Combination with pruned auxiliary
-----------------------------------
Uses _constrained_auxiliary_pruned (same as phase6_bnb_optimized) for
the pricing step.  Combines both improvements in one solver.

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


# ---------------------------------------------------------------------------
# Constrained pruned auxiliary — same as phase6_bnb_optimized
# ---------------------------------------------------------------------------

def _constrained_auxiliary_pruned(inst  : MSSCInstance,
                                   lam   : np.ndarray,
                                   sigma : float,
                                   node  : BranchNode):
    aux = solve_auxiliary_pruned(inst, lam, sigma)
    if (aux.cluster is not None and
            column_satisfies_constraints(aux.cluster, node)):
        return aux
    return constrained_auxiliary_general(inst, lam, sigma, node)


# ---------------------------------------------------------------------------
# Coverage helper
# ---------------------------------------------------------------------------

def _ensure_coverage(init_cols  : List[Cluster],
                     all_columns: List[Cluster],
                     node       : BranchNode,
                     n          : int) -> List[Cluster]:
    """
    Guarantee every entity 0..n-1 is covered by at least one column in
    init_cols.  For any uncovered entity, adds the first valid covering
    column found in all_columns.  Returns the (possibly extended) list.
    """
    covered = set()
    for cl in init_cols:
        covered.update(cl.indices)

    if len(covered) == n:
        return init_cols

    uncovered   = set(range(n)) - covered
    init_set    = {cl.indices for cl in init_cols}
    extra: List[Cluster] = []

    for cl in all_columns:
        if not uncovered:
            break
        if cl.indices in init_set:
            continue
        if not column_satisfies_constraints(cl, node):
            continue
        overlap = set(cl.indices) & uncovered
        if overlap:
            extra.append(cl)
            init_set.add(cl.indices)
            uncovered -= overlap

    return init_cols + extra


# ---------------------------------------------------------------------------
# Node solver with lean warm-start
# ---------------------------------------------------------------------------

def _solve_node_lean(inst        : MSSCInstance,
                     node        : BranchNode,
                     all_columns : List[Cluster],
                     warm_cols   : Optional[List[Cluster]],
                     ub          : float,
                     max_cg_iter : int  = 200,
                     verbose     : bool = False
                     ) -> Tuple[float, np.ndarray, List[Cluster], List[Cluster]]:
    """
    Solve LP relaxation at a B&B node.

    Parameters
    ----------
    warm_cols : parent's basic columns (None at root → full pool)

    Returns
    -------
    lp_obj    : float
    z_values  : np.ndarray
    new_cols  : List[Cluster]  — new columns generated at this node
    basic_cols: List[Cluster]  — columns with z>1e-4 at LP optimum
                                 (used to warm-start children)
    """
    if warm_cols is not None:
        # Lean start: filter parent's basic cols by current constraints
        init_cols = [cl for cl in warm_cols
                     if column_satisfies_constraints(cl, node)]
        # Ensure full coverage (SAME/DIFF may drop some columns)
        init_cols = _ensure_coverage(init_cols, all_columns, node, inst.n)
    else:
        # Root node or no warm-start: use entire valid pool
        init_cols = [cl for cl in all_columns
                     if column_satisfies_constraints(cl, node)]

    rmp = RestrictedMasterProblem(inst)
    if not init_cols:
        return np.inf, np.array([]), [], []
    rmp.add_columns(init_cols)

    new_cols      = []
    prev_obj      = np.inf
    no_improve_ct = 0

    for _ in range(max_cg_iter):
        lp_obj, lam, sigma = rmp.solve()

        if inst.s == 2 and node.is_root():
            aux = solve_auxiliary_2d(inst, lam, sigma)
            rc  = aux.reduced_cost
            cl  = aux.cluster
        else:
            aux = _constrained_auxiliary_pruned(inst, lam, sigma, node)
            rc  = aux.reduced_cost
            cl  = aux.cluster

        if rc >= -1e-6:
            break

        if lp_obj >= prev_obj - 1e-8:
            no_improve_ct += 1
            if no_improve_ct >= 10:
                break
        else:
            no_improve_ct = 0

        if cl is not None and column_satisfies_constraints(cl, node):
            if cl.indices not in rmp._col_index_sets:
                rmp.add_column(cl)
                new_cols.append(cl)

        prev_obj = lp_obj

    lp_obj, _, _ = rmp.solve()
    z_values      = rmp.get_solution()

    # Extract basic columns for lean warm-starting of children
    basic_cols: List[Cluster] = []
    if z_values is not None:
        basic_cols = [rmp.columns[t]
                      for t, z in enumerate(z_values)
                      if z > 1e-4]

    return (lp_obj,
            z_values if z_values is not None else np.array([]),
            new_cols,
            basic_cols)


# ---------------------------------------------------------------------------
# Lean warm-start B&B main loop
# ---------------------------------------------------------------------------

def solve_bnb_lean(inst             : MSSCInstance,
                   max_nodes        : int  = 200,
                   n_jmeans_restarts: int  = 5,
                   seed             : int  = 42,
                   verbose          : bool = True) -> BnBResult:
    """
    Exact MSSC solver: B&B with lean warm-start for child nodes.

    Same interface and return type as solve_bnb() in phase6_bnb.py.
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Lean B&B  |  {inst.name}  "
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
    # Heap items: (lb_est, node_id, node, warm_cols)
    # warm_cols=None for root → uses full pool
    heap    = [(0.0, node_id, root, None)]
    n_nodes = 0
    lp_bound = 0.0

    while heap and n_nodes < max_nodes:
        lb_est, _, node, warm_cols = heapq.heappop(heap)

        if lb_est >= ub - 1e-6:
            if verbose:
                print(f"  Node {n_nodes:3d}: PRUNED "
                      f"(lb={lb_est:.4f} >= UB={ub:.4f})")
            continue

        n_nodes += 1

        lp_obj, z_values, new_cols, basic_cols = _solve_node_lean(
            inst, node, all_columns, warm_cols, ub, verbose=False)

        for cl in new_cols:
            if cl.indices not in all_col_indices:
                all_columns.append(cl)
                all_col_indices.add(cl.indices)

        if node.is_root():
            lp_bound = lp_obj

        if verbose:
            warm_sz = len(warm_cols) if warm_cols is not None else 0
            print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                  f"LP={lp_obj:.4f}  UB={ub:.4f}  "
                  f"new_cols={len(new_cols)}  "
                  f"warm={warm_sz}→basic={len(basic_cols)}")

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

        # Pass this node's basic columns as warm-start for children
        for child in [node.branch_same(i1, i2), node.branch_diff(i1, i2)]:
            node_id += 1
            heapq.heappush(heap, (lp_obj, node_id, child, basic_cols))

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
    print("Phase 7b — Lean Warm-Start B&B Tests")
    print("=" * 60)

    # Test 1: 2D well-separated (0 branches, warm-start path unused)
    inst1 = make_synthetic_instance(n=30, k=3, s=2, spread=0.3, seed=0)
    res1  = solve_bnb_lean(inst1, verbose=True)
    assert res1.gap_pct < 1e-3, f"Expected ~0% gap, got {res1.gap_pct:.4f}%"
    print(f"\n✓ Test 1: {res1}\n")

    # Test 2: known optimal — correctness of warm-start
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
    res2 = solve_bnb_lean(inst2, verbose=False)
    assert abs(res2.optimal_cost - true_opt) < 1e-3, \
        f"Expected {true_opt:.4f}, got {res2.optimal_cost:.4f}"
    print(f"✓ Test 2: known optimal {true_opt:.4f}, found {res2.optimal_cost:.4f}")

    # Test 3: 4D — multi-dimensional, exercises general auxiliary path
    inst3 = make_synthetic_instance(n=20, k=3, s=4, spread=1.0, seed=2)
    res3  = solve_bnb_lean(inst3, verbose=False)
    assert res3.optimal_cost > 0
    print(f"✓ Test 3 (s=4): {res3}")

    # Test 4: verbose output shows warm-start sizes at child nodes
    inst4 = make_synthetic_instance(n=40, k=4, s=3, spread=0.8, seed=5)
    res4  = solve_bnb_lean(inst4, max_nodes=20, verbose=True)
    print(f"\n✓ Test 4 (branching test): {res4}")

    print("\nAll Lean B&B tests passed.")
