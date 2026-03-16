"""
Phase 1: Foundation
====================
Core data structures and utilities for MSSC (Minimum Sum-of-Squares Clustering).

Key concepts from the paper:
  - n entities at points p_i ∈ R^s
  - k clusters, each with a centroid y_j
  - Objective: minimize sum of ||p_i - y_{c(i)}||^2  over all entities
  - Optimal centroid for a fixed cluster = arithmetic mean (centroid) of its members
    (from first-order optimality conditions, equation (2) in paper)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  Core data structure
# ---------------------------------------------------------------------------

@dataclass
class MSSCInstance:
    """
    Holds a clustering instance.

    Attributes
    ----------
    points : np.ndarray  shape (n, s)
        The n entity coordinates in R^s.
    k : int
        Number of clusters requested.
    name : str
        Human-readable label (for reporting).
    """
    points: np.ndarray          # shape (n, s)
    k: int
    name: str = "unnamed"

    def __post_init__(self):
        self.n, self.s = self.points.shape

    def __repr__(self):
        return (f"MSSCInstance(name={self.name!r}, n={self.n}, "
                f"s={self.s}, k={self.k})")


@dataclass
class MSSCSolution:
    """
    Holds a (possibly partial) clustering solution.

    Attributes
    ----------
    labels : np.ndarray  shape (n,)
        labels[i] = cluster index (0-based) of entity i.
    centroids : np.ndarray  shape (k, s)
        Centroid of each cluster.
    cost : float
        Total sum of squared distances (MSSC objective value).
    """
    labels: np.ndarray
    centroids: np.ndarray
    cost: float

    def __repr__(self):
        return f"MSSCSolution(cost={self.cost:.6f})"


# ---------------------------------------------------------------------------
# 2.  Core mathematical utilities
# ---------------------------------------------------------------------------

def compute_centroid(points: np.ndarray) -> np.ndarray:
    """
    Return the centroid (arithmetic mean) of a set of points.
    This is exact due to first-order optimality (equation 2 in paper):
        y_j^r = (sum_i x_ij * p_i^r) / (sum_i x_ij)
    """
    return points.mean(axis=0)


def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two points."""
    diff = a - b
    return float(diff @ diff)


def sq_dist_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute the (n x k) matrix of squared distances.
    D[i, j] = ||p_i - y_j||^2

    Uses the identity  ||a-b||^2 = ||a||^2 - 2 a·b + ||b||^2
    for efficiency (avoids an explicit n×k×s loop).
    """
    p_sq  = (points ** 2).sum(axis=1)   # shape (n,)
    c_sq  = (centers ** 2).sum(axis=1)  # shape (k,)
    cross = points @ centers.T          # shape (n, k)
    D = p_sq[:, None] - 2 * cross + c_sq[None, :]
    np.clip(D, 0.0, None, out=D)        # numerical safety
    return D


def cluster_cost(points: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the MSSC cost for a single cluster and return its centroid.

        cost = sum_i ||p_i - centroid||^2

    Returns (cost, centroid).
    """
    if len(points) == 0:
        raise ValueError("cluster_cost called on empty set")
    centroid = compute_centroid(points)
    diffs    = points - centroid
    cost     = float((diffs ** 2).sum())
    return cost, centroid


def total_cost_from_labels(inst: MSSCInstance,
                            labels: np.ndarray) -> float:
    """
    Given an assignment vector, compute the total MSSC objective value.
    """
    total = 0.0
    for j in range(inst.k):
        members = inst.points[labels == j]
        if len(members) > 0:
            total += cluster_cost(members)[0]
    return total


def compute_centroids_from_labels(inst: MSSCInstance,
                                   labels: np.ndarray) -> np.ndarray:
    """
    Return centroids array of shape (k, s) from a label vector.
    Empty clusters get zero centroid.
    """
    centroids = np.zeros((inst.k, inst.s))
    for j in range(inst.k):
        members = inst.points[labels == j]
        if len(members) > 0:
            centroids[j] = compute_centroid(members)
    return centroids


# ---------------------------------------------------------------------------
# 3.  Cluster — the column in the set-partitioning formulation (eq. 3)
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """
    A single cluster as needed by the set-partitioning formulation (eq. 3).

    Attributes
    ----------
    indices : tuple[int, ...]
        Sorted entity indices belonging to this cluster.
    centroid : np.ndarray  shape (s,)
    cost : float
        c_t = sum_{i in S} ||p_i - centroid||^2
    """
    indices: tuple
    centroid: np.ndarray
    cost: float

    def __repr__(self):
        return f"Cluster(indices={self.indices}, cost={self.cost:.4f})"

    def __hash__(self):
        return hash(self.indices)

    def __eq__(self, other):
        return self.indices == other.indices


def make_cluster(inst: MSSCInstance, indices) -> Cluster:
    """
    Build a Cluster object from an iterable of entity indices.
    """
    idx = tuple(sorted(indices))
    pts = inst.points[list(idx)]
    cost, centroid = cluster_cost(pts)
    return Cluster(indices=idx, centroid=centroid, cost=cost)


# ---------------------------------------------------------------------------
# 4.  Dataset helpers
# ---------------------------------------------------------------------------

def load_points_from_csv(path: str,
                          delimiter: str = ',',
                          skip_header: bool = True) -> np.ndarray:
    """Load numeric point data from a CSV file."""
    data = np.genfromtxt(path, delimiter=delimiter,
                          skip_header=int(skip_header))
    return data.astype(float)


def make_synthetic_instance(n: int, k: int, s: int = 2,
                              spread: float = 1.0,
                              seed: int = 42) -> MSSCInstance:
    """
    Generate a synthetic instance with k Gaussian blobs.
    Useful for unit testing.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10 * k, size=(k, s))
    pts = []
    for c in centers:
        pts.append(c + rng.normal(0, spread, size=(n // k, s)))
    points = np.vstack(pts)
    rng.shuffle(points)
    return MSSCInstance(points=points, k=k,
                         name=f"synthetic_n{n}_k{k}_s{s}")


# ---------------------------------------------------------------------------
# 5.  Dual variable bound estimation  (Section 5, paper)
# ---------------------------------------------------------------------------

def estimate_dual_bounds(inst: MSSCInstance,
                          ub_solution: MSSCSolution
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate lower and upper bounds for dual variables λ_i
    from a given upper-bound solution UB.

    From the paper (Section 5):
      lb_i = cost reduction when entity i is removed from its cluster in UB
      ub_i = cost increase when entity i is moved to its second-closest centroid

    Returns
    -------
    lb : np.ndarray  shape (n,)
    ub : np.ndarray  shape (n,)
    """
    n         = inst.n
    labels    = ub_solution.labels
    centroids = ub_solution.centroids

    lb     = np.zeros(n)
    ub_arr = np.zeros(n)

    for i in range(n):
        p = inst.points[i]
        j = labels[i]
        members_idx = np.where(labels == j)[0]
        m           = len(members_idx)

        # lower bound: cost reduction from removing entity i
        if m == 1:
            lb[i] = 0.0
        else:
            members_pts     = inst.points[members_idx]
            cost_with, _    = cluster_cost(members_pts)
            members_without = inst.points[
                [idx for idx in members_idx if idx != i]]
            cost_without, _ = cluster_cost(members_without)
            lb[i]           = cost_with - cost_without

        # upper bound: cost increase from moving i to second-closest centroid
        dists          = np.array([sq_dist(p, centroids[jj])
                                   for jj in range(inst.k)])
        sorted_cl      = np.argsort(dists)
        second_j       = (sorted_cl[1] if sorted_cl[0] == j
                          else sorted_cl[0])
        orig_contrib   = sq_dist(p, centroids[j])
        new_contrib    = sq_dist(p, centroids[second_j])
        ub_arr[i]      = new_contrib - orig_contrib

    lb     = np.maximum(lb, 0.0)
    ub_arr = np.maximum(ub_arr, lb)
    return lb, ub_arr


# ---------------------------------------------------------------------------
# 6.  Quick sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # sq_dist_matrix
    pts  = np.array([[0., 0.], [3., 4.], [1., 1.]])
    ctrs = np.array([[0., 0.], [3., 4.]])
    D    = sq_dist_matrix(pts, ctrs)
    assert abs(D[0, 0] -  0.) < 1e-10
    assert abs(D[1, 1] -  0.) < 1e-10
    assert abs(D[0, 1] - 25.) < 1e-10
    print("✓ sq_dist_matrix")

    # cluster_cost
    pts2       = np.array([[0., 0.], [2., 0.], [1., 0.]])
    cost, cent = cluster_cost(pts2)
    assert abs(cost - 2.0) < 1e-10
    print("✓ cluster_cost")

    # Huygens' theorem
    pts3        = np.array([[0., 0.], [2., 0.], [1., 2.]])
    cost_d, _   = cluster_cost(pts3)
    n3          = len(pts3)
    huygens     = sum(sq_dist(pts3[i], pts3[j])
                      for i in range(n3)
                      for j in range(i+1, n3)) / n3
    assert abs(cost_d - huygens) < 1e-9
    print("✓ Huygens' theorem")

    # make_synthetic_instance
    inst = make_synthetic_instance(n=60, k=3, s=2)
    print(f"✓ Synthetic instance: {inst}")

    print("\nAll Phase 1 checks passed.")
