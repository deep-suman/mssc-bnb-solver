"""
Phase 2a: k-means (Lloyd's Algorithm)
=======================================
Implements properties (ii) and (iii) from the paper (page 196):
  (ii)  Given centroids → assign each entity to nearest centroid
  (iii) Given assignments → recompute centroids as arithmetic means (eq. 2)

This gives a local optimum for MSSC and is the starting point for j-means.
"""

import numpy as np
from typing import Optional
from phase1_foundation import (
    MSSCInstance, MSSCSolution,
    sq_dist_matrix, compute_centroid
)


# ---------------------------------------------------------------------------
# 1.  k-means++ initialisation
# ---------------------------------------------------------------------------

def _kmeanspp_init(points: np.ndarray, k: int,
                   rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ seeding: pick k centroids with probability proportional to D².
    Ensures good spread and avoids degenerate starting points.
    """
    n, s = points.shape
    centroids = np.empty((k, s))

    # First centroid: uniform random
    centroids[0] = points[rng.integers(n)]

    for j in range(1, k):
        D        = sq_dist_matrix(points, centroids[:j])  # (n, j)
        min_d    = D.min(axis=1)                          # (n,)
        probs    = min_d / min_d.sum()
        centroids[j] = points[rng.choice(n, p=probs)]

    return centroids


# ---------------------------------------------------------------------------
# 2.  Assignment step  — property (iii) from paper
# ---------------------------------------------------------------------------

def _assign(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each entity to its nearest centroid.
    Returns labels array of shape (n,).
    """
    D = sq_dist_matrix(points, centroids)  # (n, k)
    return D.argmin(axis=1)


# ---------------------------------------------------------------------------
# 3.  Centroid update step  — property (ii) / equation (2) from paper
# ---------------------------------------------------------------------------

def _recompute_centroids(points: np.ndarray,
                          labels: np.ndarray,
                          k: int,
                          prev_centroids: np.ndarray) -> np.ndarray:
    """
    Recompute centroids as arithmetic means of assigned entities.
    Empty clusters retain their previous centroid (stability guard).
    """
    centroids = prev_centroids.copy()
    for j in range(k):
        mask = labels == j
        if mask.any():
            centroids[j] = points[mask].mean(axis=0)
    return centroids


# ---------------------------------------------------------------------------
# 4.  Cost computation
# ---------------------------------------------------------------------------

def _compute_cost(points: np.ndarray,
                  labels: np.ndarray,
                  centroids: np.ndarray) -> float:
    """
    Total MSSC cost = sum over entities of squared dist to their centroid.
    """
    D = sq_dist_matrix(points, centroids)          # (n, k)
    n = len(labels)
    return float(D[np.arange(n), labels].sum())


# ---------------------------------------------------------------------------
# 5.  k-means main function
# ---------------------------------------------------------------------------

def kmeans(inst: MSSCInstance,
           init_centroids: Optional[np.ndarray] = None,
           max_iter: int = 300,
           tol: float = 1e-9,
           seed: int = 42) -> MSSCSolution:
    """
    Lloyd's k-means algorithm.

    Parameters
    ----------
    inst           : MSSCInstance
    init_centroids : (k, s) array or None → uses k-means++ if None
    max_iter       : maximum iterations
    tol            : stop when cost change < tol
    seed           : RNG seed for k-means++ init

    Returns
    -------
    MSSCSolution  — a local optimum for MSSC
    """
    rng = np.random.default_rng(seed)

    # Initialise centroids
    if init_centroids is not None:
        centroids = init_centroids.copy()
    else:
        centroids = _kmeanspp_init(inst.points, inst.k, rng)

    labels = _assign(inst.points, centroids)
    cost   = _compute_cost(inst.points, labels, centroids)

    for _ in range(max_iter):
        centroids  = _recompute_centroids(inst.points, labels, inst.k,
                                           centroids)
        new_labels = _assign(inst.points, centroids)
        new_cost   = _compute_cost(inst.points, new_labels, centroids)

        if np.array_equal(new_labels, labels) or abs(cost - new_cost) < tol:
            labels = new_labels
            cost   = new_cost
            break

        labels, cost = new_labels, new_cost

    # Final recompute for consistency
    centroids = _recompute_centroids(inst.points, labels, inst.k, centroids)
    cost      = _compute_cost(inst.points, labels, centroids)

    return MSSCSolution(labels=labels, centroids=centroids, cost=cost)


# ---------------------------------------------------------------------------
# 6.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance

    print("=" * 50)
    print("Phase 2a — k-means Tests")
    print("=" * 50)

    # Test 1: output shapes are correct
    inst = make_synthetic_instance(n=60, k=3, s=2, seed=0)
    sol  = kmeans(inst, seed=0)
    assert sol.labels.shape    == (inst.n,),         "labels shape wrong"
    assert sol.centroids.shape == (inst.k, inst.s),  "centroids shape wrong"
    assert sol.cost >= 0,                            "cost must be non-negative"
    print(f"✓ Output shapes correct | cost = {sol.cost:.4f}")

    # Test 2: all k clusters are populated on well-separated data
    inst2 = make_synthetic_instance(n=300, k=3, s=2, spread=0.3, seed=1)
    sol2  = kmeans(inst2, seed=1)
    n_clusters = len(np.unique(sol2.labels))
    assert n_clusters == inst2.k, \
        f"Expected {inst2.k} clusters, got {n_clusters}"
    print(f"✓ All {inst2.k} clusters populated")

    # Test 3: cost is consistent with manual calculation
    from phase1_foundation import total_cost_from_labels
    manual_cost = total_cost_from_labels(inst2, sol2.labels)
    assert abs(sol2.cost - manual_cost) < 1e-6, \
        f"Cost mismatch: {sol2.cost:.6f} vs {manual_cost:.6f}"
    print(f"✓ Cost consistent: kmeans={sol2.cost:.4f} manual={manual_cost:.4f}")

    # Test 4: second run from same seed gives same result (deterministic)
    sol3 = kmeans(inst2, seed=1)
    assert abs(sol2.cost - sol3.cost) < 1e-10, "Not deterministic!"
    print(f"✓ Deterministic across runs")

    # Test 5: custom init_centroids is respected
    rng  = np.random.default_rng(99)
    idx  = rng.choice(inst2.n, inst2.k, replace=False)
    init = inst2.points[idx]
    sol4 = kmeans(inst2, init_centroids=init)
    assert sol4.cost >= 0
    print(f"✓ Custom init accepted | cost = {sol4.cost:.4f}")

    print("\nAll k-means tests passed.")
