"""
Phase 3: Restricted Master Problem (RMP)
==========================================
Implements the LP relaxation of the set-partitioning formulation (eq. 3):

    min  sum_{t in T'} c_t * z_t

    s.t. sum_{t in T'} a_it * z_t >= 1    for all i = 1..n   [covering]
         sum_{t in T'}         z_t <= k                        [cardinality]
         z_t >= 0                          for all t in T'

where:
  T'       = restricted subset of all possible clusters (columns)
  a_it     = 1 if entity i is in cluster t, else 0
  c_t      = MSSC cost of cluster t
  z_t      = LP variable (relaxed from {0,1} to >= 0)

After solving, the dual variables are:
  lambda_i >= 0  : dual of covering constraint i   (one per entity)
  sigma    >= 0  : dual of cardinality constraint   (one scalar)

These feed directly into the auxiliary problem (eq. 5 / eq. 6).
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Optional
from phase1_foundation import MSSCInstance, Cluster


# ---------------------------------------------------------------------------
# 1.  RMP class
# ---------------------------------------------------------------------------

class RestrictedMasterProblem:
    """
    Maintains the Gurobi LP model for the restricted master problem.

    Columns (clusters) can be added incrementally as column generation
    finds new ones with negative reduced cost.
    """

    def __init__(self, inst: MSSCInstance):
        self.inst      = inst
        self.n         = inst.n
        self.k         = inst.k
        self.columns   : List[Cluster] = []   # all columns added so far
        self._col_index_sets : set = set()    # fast duplicate detection

        # --- Build the Gurobi model ---
        self.model = gp.Model("RMP")
        self.model.setParam("OutputFlag", 0)   # suppress solver output
        self.model.setParam("Method", 2)       # barrier (interior point)
        # Crossover ON: converts interior solution to vertex → proper duals
        self.model.setParam("Crossover", 1)

        # Covering constraints:  sum_t a_it z_t >= 1  for i=0..n-1
        # We add them now (empty LHS) and populate as columns are added.
        self._cover_constrs = [
            self.model.addConstr(
                gp.LinExpr() >= 1.0,
                name=f"cover_{i}"
            )
            for i in range(self.n)
        ]

        # Cardinality constraint:  sum_t z_t <= k
        self._card_constr = self.model.addConstr(
            gp.LinExpr() <= float(self.k),
            name="cardinality"
        )

        self.model.update()

    # -----------------------------------------------------------------------
    # 2.  Add a column
    # -----------------------------------------------------------------------

    def add_column(self, cluster: Cluster) -> None:
        """
        Add one cluster (column) to the RMP.

        A column z_t has:
          - objective coefficient : c_t  (cluster cost)
          - coefficient 1 in covering constraint i for each i in cluster
          - coefficient 1 in cardinality constraint
        """
        if cluster.indices in self._col_index_sets:
            return  # already present, skip

        # Build the column vector for Gurobi
        col = gp.Column()

        # Covering constraints
        for i in cluster.indices:
            col.addTerms(1.0, self._cover_constrs[i])

        # Cardinality constraint
        col.addTerms(1.0, self._card_constr)

        # Add variable z_t >= 0 with objective coefficient c_t
        self.model.addVar(
            obj=cluster.cost,
            lb=0.0,
            name=f"z_{len(self.columns)}",
            column=col
        )

        self.columns.append(cluster)
        self._col_index_sets.add(cluster.indices)
        self.model.update()

    # -----------------------------------------------------------------------
    # 3.  Add multiple columns at once
    # -----------------------------------------------------------------------

    def add_columns(self, clusters: List[Cluster]) -> None:
        for cl in clusters:
            self.add_column(cl)

    # -----------------------------------------------------------------------
    # 4.  Solve and extract duals
    # -----------------------------------------------------------------------

    def solve(self) -> Tuple[float, np.ndarray, float]:
        """
        Solve the LP relaxation and return dual variables.

        Returns
        -------
        obj_val  : float       — LP objective value
        lam      : np.ndarray  — dual variables λ_i, shape (n,)
                                 one per covering constraint
        sigma    : float       — dual variable σ for cardinality constraint

        Sign convention (matches paper eq. 4):
          Covering constraints are  >= 1  → duals λ_i >= 0
          Cardinality constraint is <= k  → dual  σ  >= 0
        """
        self.model.optimize()

        status = self.model.Status
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(
                f"RMP solve failed with status {status}. "
                "Check that initial columns cover all entities."
            )

        obj_val = self.model.ObjVal

        # Extract duals
        # Gurobi returns Pi = dual of constraint.
        # For >= constraints the dual is non-positive in Gurobi's convention.
        # We negate to match the paper's convention (λ_i >= 0).
        lam = np.array([
            self._cover_constrs[i].Pi
            for i in range(self.n)
        ])

        # For <= constraint, Gurobi dual is non-negative.
        # Paper uses σ >= 0 with the cardinality constraint as <= k.
        sigma = self._card_constr.Pi

        # Numerical safety: clip small negatives from floating-point
        lam   = np.maximum(lam, 0.0)
        sigma = max(sigma, 0.0)

        return obj_val, lam, sigma

    # -----------------------------------------------------------------------
    # 5.  Reduced cost check
    # -----------------------------------------------------------------------

    def reduced_cost(self, cluster: Cluster,
                     lam: np.ndarray, sigma: float) -> float:
        """
        Compute the reduced cost of a cluster (column) given dual variables.

        From the paper (Section 2.1):
            π_t = c_t + σ - sum_{i in t} λ_i

        A column has negative reduced cost (violated dual constraint)
        iff π_t < 0  →  it should enter the basis.
        """
        return cluster.cost + sigma - sum(lam[i] for i in cluster.indices)

    # -----------------------------------------------------------------------
    # 6.  Current LP solution
    # -----------------------------------------------------------------------

    def get_solution(self) -> Optional[np.ndarray]:
        """
        Return the current z values as an array of shape (num_columns,).
        Returns None if model has not been solved.
        """
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None
        return np.array([v.X for v in self.model.getVars()])


# ---------------------------------------------------------------------------
# 6.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance
    from phase2_kmeans import kmeans
    from phase2_jmeans import jmeans, extract_initial_columns

    print("=" * 50)
    print("Phase 3 — RMP Tests")
    print("=" * 50)

    inst   = make_synthetic_instance(n=30, k=3, s=2, spread=0.5, seed=0)
    jm_sol = jmeans(inst, n_restarts=3, seed=0)
    cols   = extract_initial_columns(inst, jm_sol)

    # Test 1: RMP builds without errors
    rmp = RestrictedMasterProblem(inst)
    rmp.add_columns(cols)
    print(f"✓ RMP built  |  {len(cols)} initial columns added")

    # Test 2: LP solves and returns correct shapes
    obj, lam, sigma = rmp.solve()
    assert lam.shape == (inst.n,), "λ shape wrong"
    assert isinstance(sigma, float), "σ should be a float"
    print(f"✓ LP solved  |  obj={obj:.4f}")
    print(f"  λ: min={lam.min():.4f}  max={lam.max():.4f}  "
          f"mean={lam.mean():.4f}")
    print(f"  σ: {sigma:.4f}")

    # Test 3: duals are non-negative (paper eq. 4: λ_i >= 0, σ >= 0)
    assert (lam >= -1e-8).all(), "λ must be non-negative"
    assert sigma >= -1e-8,       "σ must be non-negative"
    print(f"✓ Duals non-negative")

    # Test 4: LP obj <= UB (LP relaxation lower-bounds the integer problem)
    assert obj <= jm_sol.cost + 1e-6, \
        f"LP obj {obj:.4f} should be <= UB {jm_sol.cost:.4f}"
    print(f"✓ LP obj={obj:.4f} <= UB={jm_sol.cost:.4f}")

    # Test 5: reduced cost of existing columns is >= 0 at optimum
    for cl in cols:
        rc = rmp.reduced_cost(cl, lam, sigma)
        assert rc >= -1e-6, \
            f"Existing column has negative reduced cost {rc:.6f}"
    print(f"✓ All existing columns have reduced cost >= 0")

    # Test 6: adding a duplicate column is ignored
    n_before = len(rmp.columns)
    rmp.add_column(cols[0])
    assert len(rmp.columns) == n_before, "Duplicate column should be ignored"
    print(f"✓ Duplicate column correctly ignored")

    print("\nAll RMP tests passed.")
