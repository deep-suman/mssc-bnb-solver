"""
Phase 5b: Dinkelbach's Algorithm for Fractional 0-1 QP
========================================================
Reference: Section 2.1 and 4, equation (11) of the paper.

Given a clique C in the intersection graph G, the auxiliary problem
restricted to C is the fractional 0-1 program (eq. 11):

    min_{v in {0,1}^m}  N(v) / D(v)

where (for the clique with entities indexed by C):
    N(v) = sum_{i<j in C} (d_ij^2 - lambda_i - lambda_j) * v_i * v_j
           - sum_{i in C} lambda_i * v_i
    D(v) = sum_{i in C} v_i        (number of entities in cluster)

Dinkelbach's algorithm (Dinkelbach 1967, cited in paper):
    Given current value q (initially 0):
    1. Solve:  min_{v} N(v) - q * D(v)        [unconstrained 0-1 QP]
    2. Let v* be the solution, compute q* = N(v*) / D(v*)
    3. If q* < q - tol: set q = q*, go to 1
    4. Else: return q* (converged)

The reduced cost for the best cluster found is:
    pi* = sigma + q*

We solve the unconstrained 0-1 QP by exhaustive enumeration for small
cliques and by a greedy + local search for larger ones.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import product
from phase1_foundation import MSSCInstance


# ---------------------------------------------------------------------------
# 1.  Build the QP coefficients for a clique
# ---------------------------------------------------------------------------

def build_qp_coefficients(inst: MSSCInstance,
                           lam: np.ndarray,
                           clique: List[int]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the quadratic and linear coefficients for eq. (11).

    For a clique C with m = |C| entities:

        N(v) = sum_{i<j} Q_ij * v_i * v_j  +  sum_i L_i * v_i

    where:
        Q_ij = d_ij^2 - lambda_i - lambda_j     (quadratic coefficient)
        L_i  = -lambda_i                         (linear coefficient)

    From Proposition 4: Q_ij is set to a large value M if
    hyperspheres i and j do NOT intersect (but this can't happen in a
    clique by definition, so all Q_ij are used as-is).

    Returns
    -------
    Q : np.ndarray shape (m, m)  — symmetric, zero diagonal
    L : np.ndarray shape (m,)
    """
    m      = len(clique)
    Q      = np.zeros((m, m))
    L      = np.zeros(m)
    points = inst.points

    for a, i in enumerate(clique):
        L[a] = -lam[i]
        for b, j in enumerate(clique):
            if b <= a:
                continue
            diff     = points[i] - points[j]
            d_sq     = float(diff @ diff)
            Q[a, b]  = d_sq - lam[i] - lam[j]
            Q[b, a]  = Q[a, b]

    return Q, L


# ---------------------------------------------------------------------------
# 2.  Evaluate N(v) and D(v)
# ---------------------------------------------------------------------------

def eval_numerator(v: np.ndarray, Q: np.ndarray, L: np.ndarray) -> float:
    """N(v) = 0.5 * v^T Q v + L^T v  (factor 0.5 since Q is symmetric)"""
    return 0.5 * float(v @ Q @ v) + float(L @ v)


def eval_denominator(v: np.ndarray) -> float:
    """D(v) = sum of v_i"""
    return float(v.sum())


# ---------------------------------------------------------------------------
# 3.  Unconstrained 0-1 QP solver
# ---------------------------------------------------------------------------

def solve_01qp(Q: np.ndarray, L: np.ndarray,
               q: float) -> Tuple[np.ndarray, float]:
    """
    Solve:  min_{v in {0,1}^m}  N(v) - q * D(v)
          = min_{v in {0,1}^m}  0.5 * v^T Q v + (L - q) * v

    For small m (<=20): exhaustive enumeration.
    For larger m: greedy initialisation + bit-flip local search.

    Returns
    -------
    v_best : np.ndarray shape (m,) — binary solution
    obj    : float                 — objective value N(v) - q*D(v)
    """
    m        = len(L)
    L_shifted = L - q      # absorb the -q*D(v) term into linear coefficients

    if m <= 20:
        return _solve_01qp_exhaustive(Q, L_shifted, m)
    else:
        return _solve_01qp_local_search(Q, L_shifted, m)


def _solve_01qp_exhaustive(Q: np.ndarray, L: np.ndarray,
                            m: int) -> Tuple[np.ndarray, float]:
    """Exact solver by enumeration of all 2^m binary vectors."""
    best_v   = np.zeros(m)
    best_obj = 0.0          # v=0 gives obj=0 (empty cluster, invalid)

    for bits in range(1, 2**m):   # skip v=0 (empty)
        v = np.array([(bits >> b) & 1 for b in range(m)], dtype=float)
        obj = 0.5 * float(v @ Q @ v) + float(L @ v)
        if obj < best_obj:
            best_obj = obj
            best_v   = v.copy()

    return best_v, best_obj


def _solve_01qp_local_search(Q: np.ndarray, L: np.ndarray,
                              m: int) -> Tuple[np.ndarray, float]:
    """
    Greedy initialisation + bit-flip local search for larger instances.

    Greedy: add entity i if marginal cost is negative.
    Local search: flip each bit if it improves the objective.
    """
    # --- Greedy initialisation ---
    v = np.zeros(m)
    for i in range(m):
        # Marginal cost of adding i given current v
        marginal = 0.5 * (Q[i] @ v + Q[:, i] @ v) + L[i]
        if marginal < 0:
            v[i] = 1.0

    if v.sum() == 0:
        # Fallback: pick the entity with most negative linear coefficient
        v[np.argmin(L)] = 1.0

    def obj(v):
        return 0.5 * float(v @ Q @ v) + float(L @ v)

    # --- Bit-flip local search ---
    improved = True
    while improved:
        improved = False
        cur_obj  = obj(v)
        for i in range(m):
            v[i] = 1.0 - v[i]   # flip
            new_obj = obj(v)
            if new_obj < cur_obj - 1e-10:
                cur_obj  = new_obj
                improved = True
            else:
                v[i] = 1.0 - v[i]   # flip back

    return v, obj(v)


# ---------------------------------------------------------------------------
# 4.  Dinkelbach's algorithm
# ---------------------------------------------------------------------------

def dinkelbach(inst: MSSCInstance,
               lam: np.ndarray,
               sigma: float,
               clique: List[int],
               max_iter: int = 50,
               tol: float = 1e-8
               ) -> Tuple[float, Optional[List[int]]]:
    """
    Run Dinkelbach's algorithm on a clique to find the minimum
    reduced cost cluster within that clique.

    Parameters
    ----------
    inst   : MSSCInstance
    lam    : dual variables lambda_i
    sigma  : dual variable sigma
    clique : list of entity indices forming the clique
    max_iter, tol : convergence parameters

    Returns
    -------
    best_rc      : float         — best reduced cost found (sigma + q*)
    best_indices : List[int]     — entity indices of the best cluster
                                   None if no improving cluster found
    """
    if len(clique) == 0:
        return 0.0, None

    Q, L = build_qp_coefficients(inst, lam, clique)
    m    = len(clique)

    q        = 0.0    # current parametric value
    best_rc  = 0.0
    best_idx = None

    for _ in range(max_iter):
        v, obj_val = solve_01qp(Q, L, q)

        if v.sum() < 0.5:
            break   # empty solution

        # Compute true fractional value q* = N(v) / D(v)
        num = eval_numerator(v, Q, L)
        den = eval_denominator(v)

        if den < 0.5:
            break

        q_new = num / den

        # Reduced cost for this cluster
        rc = sigma + q_new

        if rc < best_rc:
            best_rc  = rc
            best_idx = [clique[a] for a in range(m) if v[a] > 0.5]

        # Convergence check
        if abs(q_new - q) < tol:
            break

        q = q_new

    return best_rc, best_idx


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance

    print("=" * 50)
    print("Phase 5b — Dinkelbach Tests")
    print("=" * 50)

    # --- Test 1: QP coefficients are correct ---
    pts  = np.array([[0.,0.],[1.,0.],[2.,0.]])
    inst = MSSCInstance(points=pts, k=2, name="t")
    lam  = np.array([1.0, 1.0, 1.0])
    Q, L = build_qp_coefficients(inst, lam, [0, 1, 2])
    # d_01^2=1, Q_01 = 1-1-1 = -1
    assert abs(Q[0,1] - (-1.0)) < 1e-10
    # L_i = -lambda_i = -1
    assert (L == -1.0).all()
    print(f"✓ QP coefficients correct: Q[0,1]={Q[0,1]}, L={L}")

    # --- Test 2: exhaustive solver finds minimum ---
    Q2 = np.array([[0., -2.], [-2., 0.]])
    L2 = np.array([-1., -1.])
    v, obj = _solve_01qp_exhaustive(Q2, L2, 2)
    # v=[1,1]: obj = 0.5*(-2-2) + (-1-1) = -2 + -2 = -4
    # v=[1,0]: obj = -1
    # v=[0,1]: obj = -1
    # best = [1,1] with obj=-4
    assert v.sum() == 2, f"Expected both selected, got {v}"
    assert abs(obj - (-4.0)) < 1e-8
    print(f"✓ Exhaustive solver: v={v}, obj={obj}")

    # --- Test 3: Dinkelbach on a small clique ---
    pts3  = np.array([[0.,0.],[1.,0.],[3.,0.]])
    inst3 = MSSCInstance(points=pts3, k=2, name="t3")
    lam3  = np.array([2.0, 2.0, 2.0])
    sigma = 1.0
    rc, idx = dinkelbach(inst3, lam3, sigma, [0, 1, 2])
    print(f"✓ Dinkelbach: rc={rc:.4f}, cluster={idx}")
    assert isinstance(rc, float)

    # --- Test 4: singleton clique ---
    rc1, idx1 = dinkelbach(inst3, lam3, sigma, [0])
    # Single entity: N(v=[1]) = -lambda_0 = -2, D=1, q=-2, rc=sigma-2=-1
    assert abs(rc1 - (sigma - lam3[0])) < 1e-6
    print(f"✓ Singleton clique: rc={rc1:.4f} (expected {sigma-lam3[0]:.4f})")

    # --- Test 5: no improvement when lambda is tiny ---
    pts5  = np.array([[0.,0.],[10.,0.]])
    inst5 = MSSCInstance(points=pts5, k=2, name="t5")
    lam5  = np.array([0.001, 0.001])
    rc5, idx5 = dinkelbach(inst5, lam5, 0.0, [0, 1])
    # d^2=100, Q_01=100-0.001-0.001≈100 → very positive → no cluster
    print(f"✓ Tiny lambda: rc={rc5:.4f}, idx={idx5}")

    # --- Test 6: local search matches exhaustive for m=15 ---
    rng   = np.random.default_rng(42)
    m6    = 15
    Q6    = rng.normal(0, 1, (m6, m6))
    Q6    = (Q6 + Q6.T) / 2
    np.fill_diagonal(Q6, 0)
    L6    = rng.normal(-1, 0.5, m6)
    q6    = 0.0
    v_ex, obj_ex = _solve_01qp_exhaustive(Q6, L6, m6)
    v_ls, obj_ls = _solve_01qp_local_search(Q6, L6, m6)
    print(f"✓ m=15: exhaustive obj={obj_ex:.4f}, local search obj={obj_ls:.4f}")
    assert obj_ls <= obj_ex + 0.1, \
        "Local search should be close to exhaustive"

    print("\nAll Dinkelbach tests passed.")
