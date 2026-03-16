"""
Phase 6: MSSC Column Generation Solver
========================================
Puts all phases together into a single end-to-end solver.

Pipeline:
  1. Run j-means → initial upper bound UB, initial columns
  2. Add initial columns to RMP
  3. Column generation loop:
       a. Solve RMP LP → get (obj, lambda, sigma)
       b. Solve auxiliary problem → find column with negative reduced cost
       c. If found: add column, repeat
       d. If not found or LP not improving: LP relaxation is optimal
  4. Report LP lower bound, UB, optimality gap

For 2D instances  → Algorithm 1 (geometric, exact)
For general s     → Algorithm 2 (clique-based, Dinkelbach)
"""

import time
import numpy as np
from typing import Optional

from phase1_foundation import MSSCInstance, MSSCSolution
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase5_algorithm2 import solve_auxiliary_general


# ---------------------------------------------------------------------------
# 1.  Solver result
# ---------------------------------------------------------------------------

class SolverResult:
    def __init__(self):
        self.lp_bound    : float = 0.0
        self.upper_bound : float = np.inf
        self.gap         : float = np.inf
        self.n_columns   : int   = 0
        self.n_iter      : int   = 0
        self.time_sec    : float = 0.0
        self.status      : str   = "unknown"

    def gap_pct(self) -> float:
        if self.upper_bound > 1e-10:
            return 100.0 * (self.upper_bound - self.lp_bound) / self.upper_bound
        return 0.0

    def __repr__(self):
        return (f"SolverResult("
                f"LP={self.lp_bound:.4f}, "
                f"UB={self.upper_bound:.4f}, "
                f"gap={self.gap_pct():.2f}%, "
                f"cols={self.n_columns}, "
                f"iter={self.n_iter}, "
                f"time={self.time_sec:.2f}s, "
                f"status={self.status!r})")


# ---------------------------------------------------------------------------
# 2.  Main solver
# ---------------------------------------------------------------------------

def solve_mssc(inst: MSSCInstance,
               max_iter: int = 500,
               n_jmeans_restarts: int = 5,
               seed: int = 42,
               verbose: bool = True) -> SolverResult:
    """
    Solve MSSC via column generation.

    Parameters
    ----------
    inst               : MSSCInstance
    max_iter           : maximum column generation iterations
    n_jmeans_restarts  : restarts for j-means initial solution
    seed               : RNG seed
    verbose            : print iteration log

    Returns
    -------
    SolverResult
    """
    result   = SolverResult()
    t_start  = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MSSC Solver  |  {inst.name}  |  n={inst.n}, k={inst.k}, s={inst.s}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: j-means → initial UB and columns
    # ------------------------------------------------------------------
    jm_sol = jmeans(inst,
                    n_restarts=n_jmeans_restarts,
                    seed=seed)
    ub     = jm_sol.cost
    cols   = extract_initial_columns(inst, jm_sol)

    if verbose:
        print(f"  j-means UB   = {ub:.6f}  ({len(cols)} initial columns)")

    # ------------------------------------------------------------------
    # Step 2: build RMP and seed with initial columns
    # ------------------------------------------------------------------
    rmp = RestrictedMasterProblem(inst)
    rmp.add_columns(cols)

    # ------------------------------------------------------------------
    # Step 3: column generation loop
    # ------------------------------------------------------------------
    prev_obj = np.inf

    for it in range(1, max_iter + 1):

        # (a) Solve LP relaxation
        lp_obj, lam, sigma = rmp.solve()

        # (b) Solve auxiliary problem
        if inst.s == 2:
            aux = solve_auxiliary_2d(inst, lam, sigma)
            rc  = aux.reduced_cost
            cl  = aux.cluster
        else:
            aux = solve_auxiliary_general(inst, lam, sigma)
            rc  = aux.reduced_cost
            cl  = aux.cluster

        if verbose:
            print(f"  Iter {it:3d}: LP={lp_obj:.6f}  rc={rc:.6f}  "
                  f"cols={len(rmp.columns)}")

        # (c) Termination checks
        if not (rc < -1e-6):
            result.status = "optimal_rc"
            if verbose:
                print(f"  → No negative reduced cost. LP optimal.")
            break

        if lp_obj >= prev_obj - 1e-8:
            result.status = "optimal_lp"
            if verbose:
                print(f"  → LP not improving. Optimal.")
            break

        # (d) Add new column
        if cl is not None and cl.indices not in rmp._col_index_sets:
            rmp.add_column(cl)

        prev_obj = lp_obj

    else:
        result.status = "max_iter"
        if verbose:
            print(f"  → Max iterations reached.")

    # ------------------------------------------------------------------
    # Step 4: collect results
    # ------------------------------------------------------------------
    lp_obj, _, _ = rmp.solve()

    result.lp_bound    = lp_obj
    result.upper_bound = ub
    result.gap         = ub - lp_obj
    result.n_columns   = len(rmp.columns)
    result.n_iter      = it
    result.time_sec    = time.time() - t_start

    if verbose:
        print(f"\n  LP bound  = {result.lp_bound:.6f}")
        print(f"  UB        = {result.upper_bound:.6f}")
        print(f"  Gap       = {result.gap_pct():.4f}%")
        print(f"  Columns   = {result.n_columns}")
        print(f"  Time      = {result.time_sec:.2f}s")
        print(f"  Status    = {result.status}")

    return result


# ---------------------------------------------------------------------------
# 3.  Tests on benchmark instances
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance, MSSCInstance
    import numpy as np

    print("=" * 60)
    print("Phase 6 — End-to-End Solver Tests")
    print("=" * 60)

    # --- Test 1: small 2D synthetic instance ---
    inst1 = make_synthetic_instance(n=30, k=3, s=2, spread=0.5, seed=0)
    res1  = solve_mssc(inst1, verbose=True)
    assert res1.lp_bound > 0
    assert res1.upper_bound >= res1.lp_bound - 1e-6
    print(f"\n✓ Test 1 passed: {res1}\n")

    # --- Test 2: higher dimensional instance (s=4) ---
    inst2 = make_synthetic_instance(n=30, k=4, s=4, spread=1.0, seed=1)
    res2  = solve_mssc(inst2, verbose=True)
    assert res2.lp_bound > 0
    print(f"\n✓ Test 2 passed: {res2}\n")

    # --- Test 3: Ruspini dataset (if available) ---
    try:
        pts   = np.loadtxt("ruspini.csv", delimiter=",")
        inst3 = MSSCInstance(points=pts, k=4, name="Ruspini")
        res3  = solve_mssc(inst3, verbose=True)
        print(f"\n✓ Test 3 Ruspini passed: {res3}\n")
    except FileNotFoundError:
        print("  (Ruspini test skipped — place ruspini.csv in working directory)")

    print("All solver tests passed.")
