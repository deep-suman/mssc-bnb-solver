"""
Phase 6c: Branch-and-Bound Solver
====================================
Reference: Section 2 of the paper.

Full exact MSSC solver combining:
  - Column generation (LP relaxation via Gurobi)
  - Ryan-Foster branching
  - Algorithm 1 (2D) or Algorithm 2 (general) for auxiliary problem
  - Constrained auxiliary problem at each B&B node

B&B strategy:
  - Best-first search (expand node with lowest LP lower bound)
  - Prune nodes where LP lower bound >= current UB
  - Integer solution found when LP solution is integer

At each node:
  1. Build RMP with columns filtered by branching constraints
  2. Run column generation to solve LP relaxation
  3. If LP >= UB: prune
  4. If LP solution is integer: update UB, save solution
  5. Else: find Ryan-Foster pair, create two child nodes
"""

import numpy as np
import time
import heapq
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from phase1_foundation import MSSCInstance, MSSCSolution, Cluster, make_cluster
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase5_algorithm2 import solve_auxiliary_general
from phase6_branching import (BranchNode, find_branching_pair,
                               is_integer_solution,
                               column_satisfies_constraints)
from phase6_constrained_aux import constrained_auxiliary_general


# ---------------------------------------------------------------------------
# 1.  B&B result
# ---------------------------------------------------------------------------

@dataclass
class BnBResult:
    optimal_cost  : float
    optimal_labels: Optional[np.ndarray]
    lp_bound      : float
    gap_pct       : float
    n_nodes       : int
    n_columns     : int
    time_sec      : float
    status        : str

    def __repr__(self):
        return (f"BnBResult(cost={self.optimal_cost:.4f}, "
                f"LP={self.lp_bound:.4f}, "
                f"gap={self.gap_pct:.4f}%, "
                f"nodes={self.n_nodes}, "
                f"time={self.time_sec:.2f}s, "
                f"status={self.status!r})")


# ---------------------------------------------------------------------------
# 2.  Solve one B&B node via column generation
# ---------------------------------------------------------------------------

def _solve_node(inst        : MSSCInstance,
                node        : BranchNode,
                all_columns : List[Cluster],
                ub          : float,
                max_cg_iter : int = 200,
                verbose     : bool = False
                ) -> Tuple[float, np.ndarray, List[Cluster]]:
    """
    Solve the LP relaxation at a B&B node via column generation.

    Parameters
    ----------
    inst        : MSSCInstance
    node        : current B&B node (holds branching constraints)
    all_columns : all columns generated so far (warm start)
    ub          : current upper bound (for pruning)
    max_cg_iter : column generation iteration limit

    Returns
    -------
    lp_obj   : float        — LP lower bound at this node
    z_values : np.ndarray   — LP solution values
    new_cols : List[Cluster] — new columns generated at this node
    """
    # Filter columns that satisfy branching constraints
    valid_cols = [cl for cl in all_columns
                  if column_satisfies_constraints(cl, node)]

    # Build RMP with valid columns
    rmp = RestrictedMasterProblem(inst)
    if len(valid_cols) > 0:
        rmp.add_columns(valid_cols)
    else:
        return np.inf, np.array([]), []

    new_cols      = []
    prev_obj      = np.inf
    no_improve_ct = 0       # consecutive non-improving iterations

    for _ in range(max_cg_iter):
        lp_obj, lam, sigma = rmp.solve()


        # Solve auxiliary problem with branching constraints
        if inst.s == 2 and node.is_root():
            aux = solve_auxiliary_2d(inst, lam, sigma)
            rc  = aux.reduced_cost
            cl  = aux.cluster
        else:
            aux = constrained_auxiliary_general(inst, lam, sigma, node)
            rc  = aux.reduced_cost
            cl  = aux.cluster

        # Primary termination: no negative reduced cost → LP optimal
        if rc >= -1e-6:
            break

        # Secondary termination: too many consecutive non-improving iters
        # (guards against cycling when LP is degenerate)
        if lp_obj >= prev_obj - 1e-8:
            no_improve_ct += 1
            if no_improve_ct >= 10:
                break
        else:
            no_improve_ct = 0

        # Add new column if valid and not duplicate
        if cl is not None and column_satisfies_constraints(cl, node):
            if cl.indices not in rmp._col_index_sets:
                rmp.add_column(cl)
                new_cols.append(cl)

        prev_obj = lp_obj

    lp_obj, _, _ = rmp.solve()
    z_values     = rmp.get_solution()

    return lp_obj, z_values if z_values is not None else np.array([]), new_cols


# ---------------------------------------------------------------------------
# 3.  Extract solution labels from integer LP solution
# ---------------------------------------------------------------------------

def _extract_labels(inst     : MSSCInstance,
                    columns  : List[Cluster],
                    z_values : np.ndarray) -> Optional[np.ndarray]:
    """
    Given an integer LP solution, extract cluster labels for all entities.
    Returns None if solution is infeasible.
    """
    labels = np.full(inst.n, -1, dtype=int)
    tol    = 1e-4

    for t, z in enumerate(z_values):
        if z > 1.0 - tol:   # z_t = 1
            for i in columns[t].indices:
                if labels[i] == -1:
                    labels[i] = t

    if (labels == -1).any():
        return None
    return labels


# ---------------------------------------------------------------------------
# 4.  Branch-and-bound main loop
# ---------------------------------------------------------------------------

def solve_bnb(inst            : MSSCInstance,
              max_nodes        : int   = 200,
              n_jmeans_restarts: int   = 5,
              seed             : int   = 42,
              verbose          : bool  = True) -> BnBResult:
    """
    Exact MSSC solver via branch-and-bound + column generation.

    Parameters
    ----------
    inst              : MSSCInstance
    max_nodes         : B&B node limit
    n_jmeans_restarts : j-means restarts for initial UB
    seed              : RNG seed
    verbose           : print B&B log
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  B&B Solver  |  {inst.name}  "
              f"|  n={inst.n}, k={inst.k}, s={inst.s}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: j-means initial upper bound
    # ------------------------------------------------------------------
    jm_sol      = jmeans(inst, n_restarts=n_jmeans_restarts, seed=seed)
    ub          = jm_sol.cost
    best_labels = jm_sol.labels
    all_columns = extract_initial_columns(inst, jm_sol)

    if verbose:
        print(f"  Initial UB = {ub:.6f}")

    # ------------------------------------------------------------------
    # Step 2: B&B with best-first search
    # Priority queue: (lb, node_id, node)
    # ------------------------------------------------------------------
    root      = BranchNode()
    node_id   = 0
    heap      = [(0.0, node_id, root)]
    n_nodes   = 0
    lp_bound  = 0.0

    while heap and n_nodes < max_nodes:
        lb_est, _, node = heapq.heappop(heap)

        # Prune if estimate >= UB
        if lb_est >= ub - 1e-6:
            if verbose:
                print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                      f"PRUNED (lb_est={lb_est:.4f} >= UB={ub:.4f})")
            continue

        n_nodes += 1

        # Solve LP at this node
        lp_obj, z_values, new_cols = _solve_node(
            inst, node, all_columns, ub, verbose=False)

        # Add new columns to global pool
        for cl in new_cols:
            if cl not in all_columns:
                all_columns.append(cl)

        # Update global LP bound (tightest at root)
        if node.is_root():
            lp_bound = lp_obj

        if verbose:
            print(f"  Node {n_nodes:3d} (depth={node.depth}): "
                  f"LP={lp_obj:.4f}  UB={ub:.4f}  "
                  f"new_cols={len(new_cols)}")

        if lp_obj >= ub - 1e-6:
            if verbose:
                print(f"         → PRUNED")
            continue

        # Check integrality
        if len(z_values) > 0 and is_integer_solution(z_values):
            # Integer solution found — update UB
            valid_cols = [cl for cl in all_columns
                          if column_satisfies_constraints(cl, node)]
            labels = _extract_labels(inst, valid_cols, z_values)
            if labels is not None:
                if lp_obj < ub - 1e-6:
                    ub          = lp_obj
                    best_labels = labels
                    if verbose:
                        print(f"         → NEW UB = {ub:.6f}")
            continue

        # Branch: find Ryan-Foster pair
        if len(z_values) == 0:
            continue

        valid_cols = [cl for cl in all_columns
                      if column_satisfies_constraints(cl, node)]
        pair = find_branching_pair(valid_cols, z_values)

        if pair is None:
            continue

        i1, i2 = pair
        if verbose:
            print(f"         → BRANCH on ({i1},{i2})")

        # Create two child nodes
        for child in [node.branch_same(i1, i2),
                      node.branch_diff(i1, i2)]:
            node_id += 1
            heapq.heappush(heap, (lp_obj, node_id, child))

    # ------------------------------------------------------------------
    # Step 3: collect results
    # ------------------------------------------------------------------
    elapsed  = time.time() - t_start
    status   = "optimal" if not heap or n_nodes < max_nodes else "max_nodes"
    gap      = 100.0 * (ub - lp_bound) / ub if ub > 1e-10 else 0.0

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
        status         = status
    )


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance, MSSCInstance
    import numpy as np

    print("=" * 60)
    print("Phase 6c — Branch-and-Bound Tests")
    print("=" * 60)

    # --- Test 1: well-separated 2D instance (should need 0 branches) ---
    inst1 = make_synthetic_instance(n=30, k=3, s=2, spread=0.3, seed=0)
    res1  = solve_bnb(inst1, verbose=True)
    assert res1.gap_pct < 1e-3, f"Expected 0% gap, got {res1.gap_pct:.4f}%"
    print(f"\n✓ Test 1: {res1}\n")

    # --- Test 2: known optimal synthetic instance ---
    centers = np.array([[0.,0.],[50.,0.],[25.,43.]])
    spread  = np.tile([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]], (3,1))
    points  = np.vstack([centers[c] + spread[c*5:(c+1)*5]
                         for c in range(3)])
    # recompute correctly: each center gets its own 5-point spread
    points  = np.vstack([
        centers[0] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
        centers[1] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
        centers[2] + np.array([[-2,-1],[2,-1],[0,2],[-1,1],[1,-1]]),
    ])
    inst2    = MSSCInstance(points=points, k=3, name="known_opt")
    true_opt = sum(
        ((points[c*5:(c+1)*5] -
          points[c*5:(c+1)*5].mean(axis=0))**2).sum()
        for c in range(3)
    )
    res2 = solve_bnb(inst2, verbose=False)
    assert abs(res2.optimal_cost - true_opt) < 1e-3, \
        f"Expected {true_opt:.4f}, got {res2.optimal_cost:.4f}"
    print(f"✓ Test 2: known optimal {true_opt:.4f}, "
          f"found {res2.optimal_cost:.4f}")

    # --- Test 3: higher dimensional instance ---
    inst3 = make_synthetic_instance(n=20, k=3, s=4, spread=1.0, seed=2)
    res3  = solve_bnb(inst3, verbose=False)
    assert res3.optimal_cost > 0
    print(f"✓ Test 3 (s=4): {res3}")

    print("\nAll B&B tests passed.")
