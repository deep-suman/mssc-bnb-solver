"""
Phase 8: Fast B&B using vectorised j-means initialisation.
===========================================================
Drop-in replacement for solve_bnb() in phase6_bnb.py.
Uses fast_jmeans() instead of jmeans() for the initial upper bound.
"""

import numpy as np
import time
import heapq
from typing import Optional, List, Tuple

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase2_kmeans import kmeans
from phase2_jmeans import extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase5_algorithm2_pruned import solve_auxiliary_pruned
from phase6_branching import (BranchNode, find_branching_pair,
                               is_integer_solution,
                               column_satisfies_constraints)
from phase6_constrained_aux import constrained_auxiliary_general
from phase6_bnb import BnBResult, _extract_labels, _solve_node
from phase8_fast_jmeans import fast_jmeans


def solve_bnb_fast(inst: MSSCInstance,
                   max_nodes: int = 200,
                   n_jmeans_restarts: int = 5,
                   seed: int = 42,
                   verbose: bool = True) -> BnBResult:
    """
    Exact MSSC solver: B&B with fast j-means initialisation.

    Same interface and return type as solve_bnb() in phase6_bnb.py.

    Note on UB quality
    ------------------
    fast_jmeans uses adaptive screening depth: 2-step estimates when
    n/k >= 7 (large clusters -- tight estimates needed to catch near-miss
    improving moves), 1-step otherwise (small clusters -- 1 step is already
    tight, and avoids tensor-computation overhead that dominates at high k).
    On Glass k=30 the 2-step version found 63.2478, strictly better than
    the original paper's reported 63.3284.
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Fast B&B  |  {inst.name}  |  n={inst.n}, k={inst.k}, s={inst.s}")
        print(f"{'='*60}")

    # Initial upper bound via fast j-means
    jm_sol      = fast_jmeans(inst, n_restarts=n_jmeans_restarts, seed=seed)
    ub          = jm_sol.cost
    best_labels = jm_sol.labels
    all_columns = extract_initial_columns(inst, jm_sol)

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
                print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                      f"PRUNED (lb={lb_est:.4f} >= UB={ub:.4f})")
            continue

        n_nodes += 1

        lp_obj, z_values, new_cols = _solve_node(
            inst, node, all_columns, ub, verbose=False)

        for cl in new_cols:
            if cl not in all_columns:
                all_columns.append(cl)

        if node.is_root():
            lp_bound = lp_obj

        if verbose:
            print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                  f"LP={lp_obj:.4f}  UB={ub:.4f}  new_cols={len(new_cols)}")

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
    status  = "optimal" if not heap or n_nodes < max_nodes else "max_nodes"
    gap     = 100.0 * (ub - lp_bound) / ub if ub > 1e-10 else 0.0

    if verbose:
        print(f"\n  Optimal cost = {ub:.6f}")
        print(f"  LP bound     = {lp_bound:.6f}")
        print(f"  Gap          = {gap:.4f}%")
        print(f"  B&B nodes    = {n_nodes}")
        print(f"  Columns      = {len(all_columns)}")
        print(f"  Time         = {elapsed:.2f}s")
        print(f"  Status       = {status}")

    return BnBResult(
        optimal_cost   = ub,
        optimal_labels = best_labels,
        lp_bound       = lp_bound,
        gap_pct        = gap,
        n_nodes        = n_nodes,
        n_columns      = len(all_columns),
        time_sec       = elapsed,
        status         = status,
    )
