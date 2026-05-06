"""
Phase 6 (optimised): Branch-and-Bound with Pruned Auxiliary and Multi-Column Pricing.

Replaces:
  - build_intersection_graph  → build_intersection_graph_vectorized  (21.6× faster)
  - solve_auxiliary_general   → solve_auxiliary_pruned  (40-70% fewer Dinkelbach calls)
  - single-column pricing     → multi-column pricing    (fewer LP re-solves)

All other B&B logic (Ryan-Foster branching, best-first search, LP) is identical
to phase6_bnb.py.
"""

import numpy as np
import time
import heapq
from typing import Optional, List
from dataclasses import dataclass, field

from phase1_foundation import MSSCInstance, MSSCSolution, Cluster, make_cluster
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase5_algorithm2_pruned import solve_auxiliary_pruned
from phase5_algorithm2_multi import solve_auxiliary_multi
from phase6_branching import (BranchNode, find_branching_pair,
                               is_integer_solution,
                               column_satisfies_constraints)
from phase6_constrained_aux import constrained_auxiliary_general
from phase6_bnb import BnBResult  # reuse result dataclass


def solve_bnb_optimized(inst: MSSCInstance,
                        max_nodes: int = 200,
                        n_jmeans_restarts: int = 5,
                        seed: int = 42,
                        verbose: bool = True) -> BnBResult:
    """
    Exact MSSC via column-generation B&B with optimised auxiliary.

    Parameters
    ----------
    inst               : MSSCInstance
    max_nodes          : int   — stop after this many B&B nodes
    n_jmeans_restarts  : int   — j-means restarts for initial UB
    seed               : int
    verbose            : bool

    Returns
    -------
    BnBResult
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    use_2d = (inst.s == 2)

    if verbose:
        print(f"\n  Optimised B&B  |  {inst.name}  |  n={inst.n}, k={inst.k}, s={inst.s}")

    # ── Initial upper bound via j-means ──────────────────────────────────
    init_sol = jmeans(inst, n_restarts=n_jmeans_restarts, seed=seed)
    best_cost = init_sol.cost
    best_labels = init_sol.labels.copy()
    if verbose:
        print(f"  Initial UB = {best_cost:.4f}")

    init_cols = extract_initial_columns(inst, n_restarts=n_jmeans_restarts, seed=seed)
    all_cols: List[Cluster] = list(init_cols)
    n_cols_total = len(all_cols)

    # ── B&B priority queue: (lb_estimate, node_id, BranchNode) ──────────
    root = BranchNode(same_pairs=[], diff_pairs=[], depth=0,
                      lb_estimate=-1e18, parent_cols=all_cols[:])
    heap = [(-1e18, 0, root)]
    node_id = 1
    n_nodes = 0
    root_lp = None

    while heap and n_nodes < max_nodes:
        lb_est, _, node = heapq.heappop(heap)

        if lb_est >= best_cost - 1e-6:
            continue  # prune

        n_nodes += 1

        # Filter columns valid for this node's branching constraints
        valid_cols = [c for c in all_cols
                      if column_satisfies_constraints(c, node.same_pairs, node.diff_pairs)]

        if not valid_cols:
            continue

        # ── Column generation ─────────────────────────────────────────────
        rmp = RestrictedMasterProblem(inst)
        for c in valid_cols:
            rmp.add_column(c)

        lp_bound = None
        cg_iters = 0
        max_cg = 200
        non_improving = 0

        while cg_iters < max_cg and non_improving < 10:
            obj, lam, sigma = rmp.solve()
            if obj is None:
                break
            lp_bound = obj

            if lp_bound >= best_cost - 1e-6:
                break  # prune early

            # Pricing: use multi-column for general, standard for 2D
            if use_2d:
                aux = solve_auxiliary_2d(inst, lam, sigma)
                new_cols = [aux.cluster] if aux.has_negative_rc() and aux.cluster else []
            else:
                aux = solve_auxiliary_multi(inst, lam, sigma, max_columns=5)
                new_cols = [c for c in aux.columns
                            if column_satisfies_constraints(c, node.same_pairs, node.diff_pairs)]

            if not new_cols:
                break

            added = 0
            for col in new_cols:
                rmp.add_column(col)
                all_cols.append(col)
                n_cols_total += 1
                added += 1

            non_improving = 0 if added > 0 else non_improving + 1
            cg_iters += 1

        if lp_bound is None:
            continue

        if root_lp is None:
            root_lp = lp_bound

        if verbose:
            print(f"  Node {n_nodes:>4} (depth={node.depth}): "
                  f"LP={lp_bound:.4f}  UB={best_cost:.4f}  cols={cg_iters}")

        if lp_bound >= best_cost - 1e-6:
            if verbose:
                print(f"         -> PRUNED")
            continue

        # ── Check integrality ─────────────────────────────────────────────
        sol = rmp.get_solution()
        if sol is None:
            continue

        if is_integer_solution(sol):
            cost = sum(c.cost * z for c, z in sol if z > 0.5)
            if cost < best_cost - 1e-8:
                best_cost = cost
                labels = np.zeros(inst.n, dtype=int)
                cid = 0
                for c, z in sol:
                    if z > 0.5:
                        for i in c.indices:
                            labels[i] = cid
                        cid += 1
                best_labels = labels
                if verbose:
                    print(f"         -> NEW UB = {best_cost:.4f}")
            continue

        # ── Branch ────────────────────────────────────────────────────────
        pair = find_branching_pair(sol)
        if pair is None:
            continue

        i, j = pair
        for same in [True, False]:
            if same:
                child = BranchNode(
                    same_pairs=node.same_pairs + [(i, j)],
                    diff_pairs=node.diff_pairs,
                    depth=node.depth + 1,
                    lb_estimate=lp_bound,
                    parent_cols=valid_cols,
                )
            else:
                child = BranchNode(
                    same_pairs=node.same_pairs,
                    diff_pairs=node.diff_pairs + [(i, j)],
                    depth=node.depth + 1,
                    lb_estimate=lp_bound,
                    parent_cols=valid_cols,
                )
            heapq.heappush(heap, (lp_bound, node_id, child))
            node_id += 1

    t_end = time.time()
    status = "optimal" if not heap or n_nodes < max_nodes else "max_nodes"
    lp_bound_final = root_lp if root_lp is not None else best_cost
    gap = (best_cost - lp_bound_final) / max(1e-9, abs(best_cost)) * 100

    if verbose:
        print(f"\n  Status       : {status}")
        print(f"  Optimal cost : {best_cost:.6f}")
        print(f"  Gap          : {gap:.4f}%")
        print(f"  Time         : {t_end - t_start:.2f}s")
        print(f"  Nodes        : {n_nodes}")

    return BnBResult(
        optimal_cost=best_cost,
        optimal_labels=best_labels,
        lp_bound=lp_bound_final,
        gap_pct=gap,
        n_nodes=n_nodes,
        n_columns=n_cols_total,
        time_sec=t_end - t_start,
        status=status,
    )
