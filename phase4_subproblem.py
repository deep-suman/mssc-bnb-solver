"""
Phase 4c: Solving Subproblem (7)
==================================
Reference: Section 3, equation (7) of the paper.

For a given index set S, subproblem (7) is:

    min_y  sum_{i in S} ||p_i - y||^2
    s.t.   ||p_i - y||^2 <= lambda_i    for all i in S

This is a convex QP (quadratic objective, convex quadratic constraints).

Key insight from the paper (Proposition 1):
    The optimal y* is the centroid of {p_i | i in S}.
    If this centroid is feasible (inside all discs in S), we are done.
    Otherwise, the optimum lies on the boundary of one or more discs.

We solve it in two steps:
    1. Check if centroid is feasible → return immediately if yes.
    2. Otherwise, use scipy.optimize.minimize with constraints.

The VALUE of the subproblem feeds into the auxiliary problem objective:
    pi_t = c_t + sigma - sum_{i in S} lambda_i
         = [sum_{i in S} ||p_i - y*||^2] + sigma - sum_{i in S} lambda_i

We return (y*, subproblem_cost) where subproblem_cost is the raw
sum of squared distances (sigma is added by the caller).
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from phase1_foundation import MSSCInstance
from phase4_geometry import Disc


# ---------------------------------------------------------------------------
# 1.  Centroid feasibility check
# ---------------------------------------------------------------------------

def _centroid_feasible(centroid: np.ndarray,
                        discs: List[Disc],
                        S: tuple) -> bool:
    """
    Check if the centroid of {p_i | i in S} lies inside all discs in S.
    i.e., ||p_i - centroid||^2 <= lambda_i  for all i in S.
    """
    for i in S:
        d    = discs[i]
        diff = centroid - d.center
        if float(diff @ diff) > d.lam + 1e-10:
            return False
    return True


# ---------------------------------------------------------------------------
# 2.  Objective and gradient
# ---------------------------------------------------------------------------

def _objective(y: np.ndarray,
               centers: np.ndarray) -> float:
    """sum_{i in S} ||p_i - y||^2"""
    diff = centers - y[None, :]    # (|S|, 2)
    return float((diff ** 2).sum())


def _gradient(y: np.ndarray,
              centers: np.ndarray) -> np.ndarray:
    """Gradient of objective w.r.t. y = -2 * sum_{i in S} (p_i - y)"""
    diff = centers - y[None, :]    # (|S|, 2)
    return -2.0 * diff.sum(axis=0)


# ---------------------------------------------------------------------------
# 3.  Solve subproblem (7)
# ---------------------------------------------------------------------------

def solve_subproblem(inst: MSSCInstance,
                     discs: List[Disc],
                     S: tuple) -> Tuple[np.ndarray, float]:
    """
    Solve subproblem (7) for index set S.

    Parameters
    ----------
    inst  : MSSCInstance
    discs : list of Disc objects (one per entity)
    S     : tuple of sorted entity indices forming the candidate cluster

    Returns
    -------
    y_opt : np.ndarray shape (2,)  — optimal cluster center
    cost  : float                  — sum_{i in S} ||p_i - y_opt||^2
    """
    if len(S) == 0:
        raise ValueError("S must be non-empty")

    centers = inst.points[list(S)]    # shape (|S|, 2)

    # --- Step 1: try the centroid (unconstrained optimum) ---
    centroid = centers.mean(axis=0)

    if _centroid_feasible(centroid, discs, S):
        diff = centers - centroid[None, :]
        cost = float((diff ** 2).sum())
        return centroid, cost

    # --- Step 2: constrained optimisation ---
    # Constraints: ||p_i - y||^2 <= lambda_i  for all i in S
    # Written as: lambda_i - ||p_i - y||^2 >= 0
    constraints = []
    for i in S:
        d = discs[i]
        def make_constr(center=d.center, lam=d.lam):
            return {
                'type': 'ineq',
                'fun' : lambda y: lam - float((y - center) @ (y - center)),
                'jac' : lambda y: 2.0 * (y - center)
            }
        constraints.append(make_constr())

    # Try multiple starting points for robustness:
    #   1. centroid (unconstrained optimum, may be infeasible)
    #   2. center of each disc in S (always feasible for its own constraint)
    # Pick the best feasible result.
    start_points = [centroid] + [discs[i].center for i in S]

    best_y    = None
    best_cost = np.inf

    for x0 in start_points:
        result = minimize(
            fun=_objective,
            x0=x0,
            jac=_gradient,
            args=(centers,),
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 2000, 'eps': 1e-10}
        )
        if not result.success:
            continue
        # Verify feasibility
        y_cand = result.x
        feasible = all(
            float((y_cand - discs[i].center) @ (y_cand - discs[i].center))
            <= discs[i].lam + 1e-6
            for i in S
        )
        if feasible and result.fun < best_cost:
            best_cost = result.fun
            best_y    = y_cand

    # Fallback: if no feasible solution found, return center of smallest disc
    if best_y is None:
        smallest = min(S, key=lambda i: discs[i].lam)
        best_y   = discs[smallest].center.copy()
        diff     = centers - best_y[None, :]
        best_cost = float((diff ** 2).sum())

    diff = centers - best_y[None, :]
    cost = float((diff ** 2).sum())

    return best_y, cost


# ---------------------------------------------------------------------------
# 4.  Reduced cost of a subproblem solution
# ---------------------------------------------------------------------------

def subproblem_reduced_cost(cost: float,
                             sigma: float,
                             lam: np.ndarray,
                             S: tuple) -> float:
    """
    Compute the reduced cost (violation) for the cluster defined by S.

    From paper eq. (5) / Section 2.1:
        pi_t = c_t + sigma - sum_{i in S} lambda_i

    If pi_t < 0, this cluster has negative reduced cost and should
    be added to the master problem.
    """
    return cost + sigma - sum(lam[i] for i in S)


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase4_geometry import build_discs
    from phase1_foundation import make_synthetic_instance

    print("=" * 50)
    print("Phase 4c — Subproblem (7) Tests")
    print("=" * 50)

    inst = make_synthetic_instance(n=10, k=3, s=2, seed=0)

    # Use large lambda so centroid is always feasible
    lam_large = np.full(inst.n, 1e6)
    discs_large = build_discs(inst.points, lam_large)

    # --- Test 1: centroid is optimal when unconstrained ---
    S = (0, 1, 2)
    y_opt, cost = solve_subproblem(inst, discs_large, S)
    centroid_manual = inst.points[list(S)].mean(axis=0)
    assert np.allclose(y_opt, centroid_manual, atol=1e-6), \
        f"Expected centroid, got {y_opt}"
    print(f"✓ Unconstrained: y* = centroid = {np.round(y_opt,4)}")

    # --- Test 2: cost matches manual calculation ---
    diff = inst.points[list(S)] - y_opt
    cost_manual = float((diff**2).sum())
    assert abs(cost - cost_manual) < 1e-8
    print(f"✓ Cost matches manual: {cost:.6f}")

    # --- Test 3: constrained case — discs intersect but centroid is outside one ---
    # p0=(0,0) r=1, p1=(1,0) r=0.2  → discs intersect (|r0-r1|=0.8 < d=1.0 < r0+r1=1.2)
    # centroid=(0.5,0): dist to p1=0.5, 0.5^2=0.25 > 0.04=lam1 → centroid outside disc 1
    pts_c = np.array([[0.0, 0.0], [1.0, 0.0]])
    lam_c = np.array([1.0, 0.04])
    discs_c = build_discs(pts_c, lam_c)
    inst_c  = MSSCInstance(points=pts_c, k=2, name="constrained")

    y_opt2, cost2 = solve_subproblem(inst_c, discs_c, (0, 1))
    # y* must be inside both discs (use 1e-4 tolerance for numerical solver)
    for i in (0, 1):
        diff = y_opt2 - discs_c[i].center
        assert float(diff @ diff) <= discs_c[i].lam + 1e-4, \
            f"y* not inside disc {i}: {float(diff@diff):.6f} > {discs_c[i].lam:.6f}"
    print(f"✓ Constrained: y*={np.round(y_opt2,4)}, cost={cost2:.4f}")

    # --- Test 4: singleton S — y* must equal p_i (only feasible point) ---
    pts_s = np.array([[3.0, 4.0]])
    lam_s = np.array([0.0])    # zero radius disc
    inst_s = MSSCInstance(points=pts_s, k=1, name="single")
    discs_s = build_discs(pts_s, lam_s)
    y_s, cost_s = solve_subproblem(inst_s, discs_s, (0,))
    assert np.allclose(y_s, pts_s[0], atol=1e-6)
    assert abs(cost_s) < 1e-8
    print(f"✓ Singleton S: y*={y_s}, cost={cost_s:.6f}")

    # --- Test 5: reduced cost calculation ---
    lam_test  = np.array([2.0, 3.0, 1.5])
    sigma     = 0.5
    S_test    = (0, 1, 2)
    rc = subproblem_reduced_cost(10.0, sigma, lam_test, S_test)
    expected  = 10.0 + 0.5 - (2.0 + 3.0 + 1.5)
    assert abs(rc - expected) < 1e-10
    print(f"✓ Reduced cost: {rc:.4f} (expected {expected:.4f})")

    print("\nAll subproblem tests passed.")
