"""
Phase 4a: Disc Intersection Geometry
======================================
Reference: Section 3 of the paper.

Each entity i defines a disc:
    D_i = { y in R^2 | ||p_i - y||^2 <= lambda_i }

with center p_i and radius sqrt(lambda_i).

Two discs D_i and D_j:
  - INTERSECT (in two points) if:  |sqrt(lambda_i) - sqrt(lambda_j)| < ||p_i - p_j|| <= sqrt(lambda_i) + sqrt(lambda_j)
  - ONE CONTAINS THE OTHER if:     ||p_i - p_j|| <= |sqrt(lambda_i) - sqrt(lambda_j)|
  - ARE DISJOINT if:               ||p_i - p_j|| > sqrt(lambda_i) + sqrt(lambda_j)

This file implements:
  1. Disc dataclass
  2. Relation check between two discs (intersect / contained / disjoint)
  3. Computing the two intersection points of two intersecting discs
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


# ---------------------------------------------------------------------------
# 1.  Disc dataclass
# ---------------------------------------------------------------------------

@dataclass
class Disc:
    """
    A disc in R^2 corresponding to entity i.

    Attributes
    ----------
    idx    : entity index i
    center : p_i, shape (2,)
    lam    : lambda_i (dual variable) — radius^2
    radius : sqrt(lambda_i)
    """
    idx    : int
    center : np.ndarray   # shape (2,)
    lam    : float        # lambda_i = radius^2

    @property
    def radius(self) -> float:
        return float(np.sqrt(max(self.lam, 0.0)))

    def contains_point(self, y: np.ndarray) -> bool:
        """True if y is inside or on the boundary of this disc."""
        diff = y - self.center
        return float(diff @ diff) <= self.lam + 1e-10

    def __repr__(self):
        return (f"Disc(idx={self.idx}, "
                f"center={np.round(self.center,2)}, "
                f"radius={self.radius:.4f})")


# ---------------------------------------------------------------------------
# 2.  Relation between two discs
# ---------------------------------------------------------------------------

class DiscRelation(Enum):
    DISJOINT   = "disjoint"       # no overlap
    INTERSECT  = "intersect"      # boundary crosses in exactly 2 points
    CONTAINED  = "contained"      # one disc entirely inside the other
    IDENTICAL  = "identical"      # same center and radius (degenerate)


def disc_relation(di: Disc, dj: Disc,
                  tol: float = 1e-10) -> DiscRelation:
    """
    Classify the spatial relationship between two discs.

    From paper (Section 3):
      d = ||p_i - p_j||
      ri = sqrt(lambda_i),  rj = sqrt(lambda_j)

      INTERSECT  iff  |ri - rj| < d <= ri + rj
      CONTAINED  iff  d <= |ri - rj|
      DISJOINT   iff  d > ri + rj
    """
    diff = di.center - dj.center
    d    = float(np.sqrt(diff @ diff))
    ri   = di.radius
    rj   = dj.radius

    sum_r  = ri + rj
    diff_r = abs(ri - rj)

    if d > sum_r + tol:
        return DiscRelation.DISJOINT
    elif d <= diff_r + tol:
        if d < tol and abs(ri - rj) < tol:
            return DiscRelation.IDENTICAL
        return DiscRelation.CONTAINED
    else:
        return DiscRelation.INTERSECT


# ---------------------------------------------------------------------------
# 3.  Intersection points of two discs
# ---------------------------------------------------------------------------

def disc_intersection_points(di: Disc, dj: Disc,
                              tol: float = 1e-10
                              ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the two intersection points of discs di and dj.

    Returns None if the discs do not intersect in exactly two points.

    Derivation:
      Let d = ||p_j - p_i||, with unit vector u = (p_j - p_i) / d.
      The intersection points lie on the radical axis at distance:
          a = (ri^2 - rj^2 + d^2) / (2d)    along u from p_i
          h = sqrt(ri^2 - a^2)               perpendicular to u

      The two points are:
          P1 = p_i + a*u + h*v
          P2 = p_i + a*u - h*v
      where v is the unit vector perpendicular to u in 2D.
    """
    rel = disc_relation(di, dj, tol)
    if rel != DiscRelation.INTERSECT:
        return None

    pi, pj = di.center, dj.center
    ri, rj = di.radius, dj.radius

    delta = pj - pi
    d     = float(np.sqrt(delta @ delta))

    # Unit vector along line p_i -> p_j
    u = delta / d

    # Perpendicular unit vector (rotate u by 90°)
    v = np.array([-u[1], u[0]])

    # Distance along u from p_i to the radical axis
    a = (ri**2 - rj**2 + d**2) / (2.0 * d)

    # Half-distance between the two intersection points
    h_sq = ri**2 - a**2
    if h_sq < 0.0:
        h_sq = 0.0          # numerical safety
    h = float(np.sqrt(h_sq))

    base = pi + a * u
    P1   = base + h * v
    P2   = base - h * v

    return P1, P2


# ---------------------------------------------------------------------------
# 4.  Build all discs from instance + dual variables
# ---------------------------------------------------------------------------

def build_discs(points: np.ndarray, lam: np.ndarray) -> List[Disc]:
    """
    Build a Disc for each entity i.
    Entities with lambda_i = 0 get zero-radius discs (single point).

    Parameters
    ----------
    points : shape (n, 2)
    lam    : shape (n,)  — dual variables lambda_i
    """
    assert points.shape[1] == 2, "Disc geometry requires 2D points"
    return [
        Disc(idx=i, center=points[i].copy(), lam=float(lam[i]))
        for i in range(len(points))
    ]


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Phase 4a — Disc Geometry Tests")
    print("=" * 50)

    # Test 1: disc_relation — DISJOINT
    d1 = Disc(0, np.array([0.0, 0.0]), lam=1.0)   # radius=1
    d2 = Disc(1, np.array([5.0, 0.0]), lam=1.0)   # radius=1, far away
    assert disc_relation(d1, d2) == DiscRelation.DISJOINT
    print("✓ DISJOINT detected correctly")

    # Test 2: disc_relation — INTERSECT
    d3 = Disc(2, np.array([1.5, 0.0]), lam=1.0)   # radius=1, overlaps d1
    assert disc_relation(d1, d3) == DiscRelation.INTERSECT
    print("✓ INTERSECT detected correctly")

    # Test 3: disc_relation — CONTAINED
    d4 = Disc(3, np.array([0.1, 0.0]), lam=9.0)   # radius=3, contains d1
    assert disc_relation(d1, d4) == DiscRelation.CONTAINED
    print("✓ CONTAINED detected correctly")

    # Test 4: intersection points lie on both disc boundaries
    pts = disc_intersection_points(d1, d3)
    assert pts is not None, "Should have intersection points"
    P1, P2 = pts
    # P1 and P2 must lie on boundary of d1: ||P - center1||^2 = lambda1
    err1 = abs(np.dot(P1 - d1.center, P1 - d1.center) - d1.lam)
    err2 = abs(np.dot(P1 - d3.center, P1 - d3.center) - d3.lam)
    assert err1 < 1e-8, f"P1 not on D1 boundary: err={err1:.2e}"
    assert err2 < 1e-8, f"P1 not on D3 boundary: err={err2:.2e}"
    print(f"✓ Intersection points on both boundaries")
    print(f"  P1={np.round(P1,4)}  P2={np.round(P2,4)}")

    # Test 5: disjoint discs return None
    pts2 = disc_intersection_points(d1, d2)
    assert pts2 is None
    print("✓ Disjoint discs return None")

    # Test 6: contained discs return None
    pts3 = disc_intersection_points(d1, d4)
    assert pts3 is None
    print("✓ Contained discs return None")

    # Test 7: contains_point
    assert d4.contains_point(np.array([0.0, 0.0]))   # center of d1 inside d4
    assert not d1.contains_point(np.array([5.0, 0.0]))
    print("✓ contains_point works correctly")

    # Test 8: build_discs
    pts_arr = np.array([[0.,0.],[1.,0.],[2.,0.]])
    lam_arr = np.array([1.0, 0.5, 2.0])
    discs   = build_discs(pts_arr, lam_arr)
    assert len(discs) == 3
    assert discs[2].radius == np.sqrt(2.0)
    print(f"✓ build_discs: {discs}")

    print("\nAll geometry tests passed.")
