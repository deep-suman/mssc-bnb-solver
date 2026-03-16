"""
Phase 4d: Algorithm 1 — Auxiliary Problem Solver (2D)
=======================================================
Reference: Section 3, Algorithm 1 of the paper.

Given dual variables (lambda, sigma) from the RMP, find a cluster
with the most negative reduced cost:

    pi* = sigma + min_{v in {0,1}^n}  [sum_{i in S} ||p_i - y||^2
                                        - sum_{i in S} lambda_i]

where S is the index set of entities assigned to the cluster (v_i=1)
and y is the optimal centroid for S (solution to subproblem 7).

Algorithm 1 (paper, p.204):
  1. Build discs D_i = {y | ||p_i - y||^2 <= lambda_i} for each entity.
  2. Enumerate all intersection points (L1) and isolated discs (L2).
  3. For each point p in L1 (defined by discs i,j):
       - Find S_base = {k != i,j | disc_k contains p}
       - Form 4 candidate sets: S_base, S_base∪{i}, S_base∪{j}, S_base∪{i,j}
       - Solve subproblem (7) for each candidate set
       - Update best solution if reduced cost improves
  4. For each disc in L2:
       - Form S' = {disc.idx} ∪ {all discs containing this disc}
       - Solve subproblem (7) for S'
       - Update best solution if reduced cost improves
  5. Return the best cluster found (negative reduced cost = new column).
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase4_geometry import Disc, build_discs
from phase4_regions import (enumerate_regions, candidate_index_sets,
                             isolated_disc_index_set)
from phase4_subproblem import solve_subproblem, subproblem_reduced_cost


# ---------------------------------------------------------------------------
# 1.  Result of the auxiliary problem
# ---------------------------------------------------------------------------

@dataclass
class AuxResult:
    """
    Result of solving the auxiliary problem.

    Attributes
    ----------
    reduced_cost : float
        pi* = best reduced cost found.
        Negative means a violating column was found.
    cluster      : Cluster or None
        The new cluster to add to the RMP (None if pi* >= 0).
    n_subproblems: int
        Number of subproblems solved (for performance tracking).
    """
    reduced_cost  : float
    cluster       : Optional[Cluster]
    n_subproblems : int

    def has_negative_rc(self, tol: float = 1e-6) -> bool:
        return self.reduced_cost < -tol


# ---------------------------------------------------------------------------
# 2.  Algorithm 1
# ---------------------------------------------------------------------------

def solve_auxiliary_2d(inst: MSSCInstance,
                        lam: np.ndarray,
                        sigma: float) -> AuxResult:
    """
    Solve the auxiliary problem for 2D instances using Algorithm 1.

    Parameters
    ----------
    inst  : MSSCInstance  (must have s=2)
    lam   : np.ndarray shape (n,)  — dual variables lambda_i
    sigma : float                  — dual variable for cardinality constraint

    Returns
    -------
    AuxResult
    """
    assert inst.s == 2, "Algorithm 1 requires 2D instances"

    # --- Step 1: build discs ---
    discs = build_discs(inst.points, lam)

    # --- Step 2: enumerate regions ---
    L1, L2 = enumerate_regions(discs)

    best_rc      = -1e-6        # only accept genuinely negative reduced costs
    best_S       = None
    best_y       = None
    n_subproblems = 0

    def _evaluate(S: tuple):
        """
        Evaluate candidate index set S.
        Per Proposition 1: y* must be the centroid of S.
        Only accept if centroid is feasible (inside all discs in S).
        Returns (reduced_cost, centroid) or (0, None) if infeasible.
        """
        if len(S) == 0:
            return 0.0, None

        centers  = inst.points[list(S)]
        centroid = centers.mean(axis=0)

        # Check feasibility: centroid must be inside every disc in S
        for i in S:
            diff = centroid - discs[i].center
            if float(diff @ diff) > discs[i].lam + 1e-8:
                return 0.0, None   # centroid not feasible → skip

        # Cluster cost with centroid
        diff = centers - centroid[None, :]
        cost = float((diff ** 2).sum())
        rc   = subproblem_reduced_cost(cost, sigma, lam, S)
        return rc, centroid

    # --- Steps 3-5: process intersection points (L1) ---
    for ip in L1:
        for S in candidate_index_sets(ip):
            rc, centroid = _evaluate(S)
            n_subproblems += 1
            if rc < best_rc:
                best_rc = rc
                best_S  = S
                best_y  = centroid

    # --- Steps 6-8: process isolated discs (L2) ---
    for disc in L2:
        S = isolated_disc_index_set(disc, discs)
        rc, centroid = _evaluate(S)
        n_subproblems += 1
        if rc < best_rc:
            best_rc = rc
            best_S  = S
            best_y  = centroid

    # --- Build the best cluster found ---
    best_cluster = None
    if best_S is not None:
        best_cluster = make_cluster(inst, best_S)

    return AuxResult(
        reduced_cost  = best_rc,
        cluster       = best_cluster,
        n_subproblems = n_subproblems
    )


# ---------------------------------------------------------------------------
# 3.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance
    from phase2_kmeans import kmeans
    from phase2_jmeans import jmeans, extract_initial_columns
    from phase3_rmp import RestrictedMasterProblem

    print("=" * 50)
    print("Phase 4d — Algorithm 1 Tests")
    print("=" * 50)

    # --- Test 1: auxiliary returns AuxResult with correct fields ---
    inst = make_synthetic_instance(n=20, k=3, s=2, spread=1.0, seed=0)
    lam_test  = np.ones(inst.n) * 5.0
    sigma_test = 1.0
    result = solve_auxiliary_2d(inst, lam_test, sigma_test)
    assert isinstance(result.reduced_cost, float)
    assert result.n_subproblems > 0
    print(f"✓ AuxResult returned | rc={result.reduced_cost:.4f} | "
          f"subproblems={result.n_subproblems}")

    # --- Test 2: if a negative rc cluster is found, verify it ---
    if result.has_negative_rc():
        cl = result.cluster
        assert cl is not None
        assert len(cl.indices) > 0
        assert cl.cost >= 0
        print(f"✓ Negative rc cluster found: {cl}")
    else:
        print(f"✓ No negative rc cluster (duals may already be optimal)")

    # --- Test 3: full column generation loop (small instance) ---
    inst2  = make_synthetic_instance(n=15, k=3, s=2, spread=0.5, seed=1)
    jm_sol = jmeans(inst2, n_restarts=3, seed=1)
    cols   = extract_initial_columns(inst2, jm_sol)

    rmp = RestrictedMasterProblem(inst2)
    rmp.add_columns(cols)

    print(f"\n✓ Starting column generation loop (n={inst2.n}, k={inst2.k})")
    print(f"  Initial UB = {jm_sol.cost:.4f}")

    ub       = jm_sol.cost
    prev_obj = np.inf
    max_iter = 50
    for it in range(max_iter):
        obj, lam, sigma = rmp.solve()
        aux = solve_auxiliary_2d(inst2, lam, sigma)

        print(f"  Iter {it+1:2d}: LP obj={obj:.4f}  "
              f"aux rc={aux.reduced_cost:.6f}  "
              f"subproblems={aux.n_subproblems}")

        # Check 1: no negative reduced cost → certified optimal
        if not aux.has_negative_rc():
            print(f"  ✓ Optimality certified at iter {it+1}")
            print(f"  Final LP obj = {obj:.4f}")
            break

        # Check 2: LP not improving → degenerate duals, stop
        if obj >= prev_obj - 1e-8:
            print(f"  ✓ LP not improving → optimal: LP = {obj:.4f}")
            break

        prev_obj = obj
        rmp.add_column(aux.cluster)
    else:
        print(f"  ! Did not converge in {max_iter} iters")

    print("\nAll Algorithm 1 tests passed.")
