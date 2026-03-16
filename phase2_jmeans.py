"""
Phase 2b: j-means Heuristic
=============================
Reference: Hansen & Mladenović (2001) — cited in paper page 196.

The j-means JUMP MOVE:
  - Take a k-means local optimum.
  - Try relocating each centroid y_j to the position of every entity p_i
    that is NOT currently in cluster j.
  - After each relocation, run a full k-means pass.
  - Accept the move if total cost strictly improves.
  - Repeat until no improving jump exists → j-means local optimum.

Why j-means over plain k-means?
  k-means can get stuck in poor local optima, especially for large k.
  The jump move escapes these by making a large perturbation (teleporting
  a centroid to an entity location), then re-optimising with k-means.

Role in the column generation algorithm (paper Section 5):
  - Provides the initial upper bound UB₀.
  - Seeds the restricted master problem with k good initial columns.
  - Used to estimate initial dual variable bounds lb_i, ub_i.
"""

import numpy as np
from typing import Optional
from phase1_foundation import (
    MSSCInstance, MSSCSolution,
    make_cluster, Cluster, total_cost_from_labels
)
from phase2_kmeans import kmeans, _assign, _recompute_centroids, _compute_cost


# ---------------------------------------------------------------------------
# 1.  Single jump move evaluation
# ---------------------------------------------------------------------------

def _try_jump(inst: MSSCInstance,
              centroids: np.ndarray,
              j: int,
              i: int) -> MSSCSolution:
    """
    Tentatively move centroid j to entity i, then run k-means to
    local optimum. Returns the resulting solution.

    Parameters
    ----------
    j : cluster index whose centroid is being moved
    i : entity index — new location for centroid j
    """
    trial_centroids    = centroids.copy()
    trial_centroids[j] = inst.points[i]
    return kmeans(inst, init_centroids=trial_centroids)


# ---------------------------------------------------------------------------
# 2.  j-means local search
# ---------------------------------------------------------------------------

def _jmeans_local_search(inst: MSSCInstance,
                          sol: MSSCSolution) -> MSSCSolution:
    """
    Core j-means loop starting from a given solution.

    Scans all (j, i) pairs where entity i is not in cluster j.
    Accepts the first improving jump found and restarts the scan.
    Stops when no improving jump exists.
    """
    best      = sol
    improved  = True

    while improved:
        improved = False

        for j in range(inst.k):
            # Candidate entities: those NOT in cluster j
            candidates = np.where(best.labels != j)[0]

            for i in candidates:
                trial = _try_jump(inst, best.centroids, j, i)

                if trial.cost < best.cost - 1e-10:
                    best     = trial
                    improved = True
                    break       # restart scan with new best

            if improved:
                break           # restart outer loop

    return best


# ---------------------------------------------------------------------------
# 3.  j-means with multiple restarts
# ---------------------------------------------------------------------------

def jmeans(inst: MSSCInstance,
           init_solution: Optional[MSSCSolution] = None,
           n_restarts: int = 5,
           seed: int = 42) -> MSSCSolution:
    """
    j-means heuristic with multiple k-means restarts for diversity.

    Parameters
    ----------
    inst           : MSSCInstance
    init_solution  : starting solution; runs k-means++ if None
    n_restarts     : number of independent k-means starts to try
    seed           : RNG seed

    Returns
    -------
    MSSCSolution — best j-means local optimum found
    """
    rng  = np.random.default_rng(seed)
    best = init_solution

    # If no starting solution provided, generate one with k-means
    if best is None:
        best = kmeans(inst, seed=int(rng.integers(1_000_000)))
        best = _jmeans_local_search(inst, best)

    # Additional restarts for diversity
    n = n_restarts - (init_solution is not None)
    for _ in range(n):
        start = kmeans(inst, seed=int(rng.integers(1_000_000)))
        sol   = _jmeans_local_search(inst, start)
        if sol.cost < best.cost:
            best = sol

    return best


# ---------------------------------------------------------------------------
# 4.  Extract initial columns for the master problem
# ---------------------------------------------------------------------------

def extract_initial_columns(inst: MSSCInstance,
                              sol: MSSCSolution) -> list:
    """
    Convert a j-means solution into Cluster objects —
    these become the initial columns in the restricted master problem.

    Each cluster in the partition maps to one column z_t in eq. (3).
    """
    columns = []
    seen    = set()

    for j in range(inst.k):
        idx = tuple(sorted(np.where(sol.labels == j)[0].tolist()))
        if len(idx) == 0 or idx in seen:
            continue
        seen.add(idx)
        columns.append(make_cluster(inst, idx))

    return columns


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance

    print("=" * 50)
    print("Phase 2b — j-means Tests")
    print("=" * 50)

    # Test 1: j-means cost <= k-means cost
    inst   = make_synthetic_instance(n=150, k=5, s=2, spread=1.5, seed=7)
    km_sol = kmeans(inst, seed=7)
    jm_sol = jmeans(inst, init_solution=km_sol, n_restarts=3, seed=7)
    assert jm_sol.cost <= km_sol.cost + 1e-8, \
        "j-means must not be worse than k-means"
    print(f"✓ j-means <= k-means  |  "
          f"k-means={km_sol.cost:.4f}  j-means={jm_sol.cost:.4f}")

    # Test 2: output shapes correct
    assert jm_sol.labels.shape    == (inst.n,)
    assert jm_sol.centroids.shape == (inst.k, inst.s)
    print(f"✓ Output shapes correct")

    # Test 3: extract_initial_columns gives exactly k non-empty columns
    cols = extract_initial_columns(inst, jm_sol)
    assert len(cols) == inst.k, \
        f"Expected {inst.k} columns, got {len(cols)}"
    print(f"✓ {len(cols)} columns extracted")

    # Test 4: every entity appears in exactly one column
    entity_count = np.zeros(inst.n, dtype=int)
    for cl in cols:
        for idx in cl.indices:
            entity_count[idx] += 1
    assert (entity_count == 1).all(), \
        "Each entity must appear in exactly one column"
    print(f"✓ Each entity in exactly one column")

    # Test 5: column costs sum to total solution cost
    col_cost_sum = sum(cl.cost for cl in cols)
    direct_cost  = total_cost_from_labels(inst, jm_sol.labels)
    assert abs(col_cost_sum - direct_cost) < 1e-6, \
        f"Cost mismatch: {col_cost_sum:.6f} vs {direct_cost:.6f}"
    print(f"✓ Column costs sum correctly: {col_cost_sum:.4f}")

    print("\nAll j-means tests passed.")
