"""
Phase 4b: Region Enumeration
==============================
Reference: Algorithm 1, Steps 1-2 (Section 3 of the paper).

Each pair of intersecting discs produces two intersection points.
Each intersection point is a potential vertex of a region — and each
region corresponds to a distinct index set S (the set of discs whose
interior contains that point).

We also handle isolated discs (list L2 in the paper): discs whose
boundary does not intersect any other disc boundary. These delimit
regions that Algorithm 1 must also check.

This file implements:
  1. Enumerate all intersection points (list L1)
  2. Enumerate all isolated discs   (list L2)
  3. For each point p in L1, find the index set S of discs containing p
"""

import numpy as np
from typing import List, Tuple, Set
from phase4_geometry import Disc, DiscRelation, disc_relation, \
                            disc_intersection_points


# ---------------------------------------------------------------------------
# 1.  Intersection point record
# ---------------------------------------------------------------------------

class IntersectionPoint:
    """
    One intersection point p between discs i and j, together with
    the set S of ALL other disc indices whose interiors contain p.

    Attributes
    ----------
    point : np.ndarray shape (2,)
    i, j  : indices of the two discs that produced this point
    S_base: set of all k != i,j such that ||p_k - p||^2 <= lambda_k
            (from Algorithm 1 step 2)
    """
    def __init__(self, point: np.ndarray, i: int, j: int):
        self.point  = point
        self.i      = i
        self.j      = j
        self.S_base : Set[int] = set()   # filled in by enumerate_regions

    def __repr__(self):
        return (f"IntPt(i={self.i}, j={self.j}, "
                f"point={np.round(self.point,4)}, "
                f"S_base={self.S_base})")


# ---------------------------------------------------------------------------
# 2.  Enumerate intersection points (L1) and isolated discs (L2)
# ---------------------------------------------------------------------------

def enumerate_regions(discs: List[Disc]
    ) -> Tuple[List[IntersectionPoint], List[Disc]]:
    """
    Algorithm 1, Steps 1-2.

    Step 1: For every pair (i,j), check if their discs intersect.
            If yes, record both intersection points in L1.
            Track which discs have at least one intersection → rest go to L2.

    Step 2: For each intersection point p in L1, find S_base =
            { k != i,j | disc_k contains p }

    Parameters
    ----------
    discs : list of Disc objects (one per entity)

    Returns
    -------
    L1 : list of IntersectionPoint
    L2 : list of Disc  (isolated — boundary intersects no other boundary)
    """
    n          = len(discs)
    L1         : List[IntersectionPoint] = []
    has_isect  = [False] * n   # True if disc i intersects at least one other

    # --- Step 1: find all pairwise intersections ---
    for i in range(n):
        for j in range(i + 1, n):
            rel = disc_relation(discs[i], discs[j])
            if rel == DiscRelation.INTERSECT:
                pts = disc_intersection_points(discs[i], discs[j])
                if pts is None:
                    continue
                P1, P2 = pts
                L1.append(IntersectionPoint(P1, i, j))
                L1.append(IntersectionPoint(P2, i, j))
                has_isect[i] = True
                has_isect[j] = True

    # --- L2: discs whose boundary intersects no other boundary ---
    L2 = [discs[i] for i in range(n) if not has_isect[i]]

    # --- Step 2: for each intersection point, find S_base ---
    for ip in L1:
        p = ip.point
        for k in range(n):
            if k == ip.i or k == ip.j:
                continue
            if discs[k].contains_point(p):
                ip.S_base.add(k)

    return L1, L2


# ---------------------------------------------------------------------------
# 3.  Build the four candidate index sets for an intersection point
# ---------------------------------------------------------------------------

def candidate_index_sets(ip: IntersectionPoint
                          ) -> List[Tuple[int, ...]]:
    """
    Algorithm 1, Step 3.

    For intersection point p defined by discs i and j, the four
    candidate index sets are (paper Section 3):
        S          = S_base
        S ∪ {i}
        S ∪ {j}
        S ∪ {i, j}

    Returns a list of tuples (sorted), duplicates removed.
    """
    base = ip.S_base
    i, j = ip.i, ip.j

    candidates = [
        base,
        base | {i},
        base | {j},
        base | {i, j},
    ]

    # Convert to sorted tuples, deduplicate, drop empty
    seen   = set()
    result = []
    for s in candidates:
        t = tuple(sorted(s))
        if len(t) > 0 and t not in seen:
            seen.add(t)
            result.append(t)

    return result


# ---------------------------------------------------------------------------
# 4.  Build the index set for an isolated disc (L2)
# ---------------------------------------------------------------------------

def isolated_disc_index_set(disc: Disc,
                             all_discs: List[Disc]
                             ) -> Tuple[int, ...]:
    """
    Algorithm 1, Steps 6-7.

    For an isolated disc D_i (in L2), the index set S' consists of:
      - disc.idx itself
      - all other disc indices j such that D_i is contained inside D_j
        (i.e., D_i ⊆ D_j)

    A point anywhere inside D_i is also inside any disc that contains D_i.
    """
    S = {disc.idx}
    for d in all_discs:
        if d.idx == disc.idx:
            continue
        # Check if disc is contained inside d:
        # condition from paper: ||p_i - p_j|| <= |r_i - r_j|  AND  r_j >= r_i
        diff   = disc.center - d.center
        dist   = float(np.sqrt(diff @ diff))
        if dist <= abs(d.radius - disc.radius) + 1e-10 and d.radius >= disc.radius:
            S.add(d.idx)
    return tuple(sorted(S))


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase4_geometry import build_discs

    print("=" * 50)
    print("Phase 4b — Region Enumeration Tests")
    print("=" * 50)

    # --- Test 1: two intersecting discs produce 2 points in L1 ---
    pts = np.array([[0.0, 0.0], [1.5, 0.0]])
    lam = np.array([1.0, 1.0])
    discs = build_discs(pts, lam)
    L1, L2 = enumerate_regions(discs)
    assert len(L1) == 2, f"Expected 2 intersection points, got {len(L1)}"
    assert len(L2) == 0, f"Expected 0 isolated discs, got {len(L2)}"
    print(f"✓ Two intersecting discs → {len(L1)} points in L1, "
          f"{len(L2)} in L2")

    # --- Test 2: two disjoint discs → L1 empty, both in L2 ---
    pts2  = np.array([[0.0, 0.0], [10.0, 0.0]])
    lam2  = np.array([1.0, 1.0])
    d2    = build_discs(pts2, lam2)
    L1b, L2b = enumerate_regions(d2)
    assert len(L1b) == 0
    assert len(L2b) == 2
    print(f"✓ Two disjoint discs → {len(L1b)} in L1, {len(L2b)} in L2")

    # --- Test 3: one disc contained in another → both in L2 ---
    pts3  = np.array([[0.0, 0.0], [0.1, 0.0]])
    lam3  = np.array([9.0, 1.0])           # big disc contains small one
    d3    = build_discs(pts3, lam3)
    L1c, L2c = enumerate_regions(d3)
    assert len(L1c) == 0
    assert len(L2c) == 2
    print(f"✓ Contained discs → {len(L1c)} in L1, {len(L2c)} in L2")

    # --- Test 4: three mutually intersecting discs ---
    pts4 = np.array([[0.0,0.0],[2.0,0.0],[1.0,1.5]])
    lam4 = np.array([2.0, 2.0, 2.0])
    d4   = build_discs(pts4, lam4)
    L1d, L2d = enumerate_regions(d4)
    # 3 pairs → up to 6 intersection points
    assert len(L1d) == 6, f"Expected 6 points, got {len(L1d)}"
    assert len(L2d) == 0
    print(f"✓ Three intersecting discs → {len(L1d)} points in L1")

    # --- Test 5: candidate_index_sets gives 4 sets ---
    ip   = L1d[0]
    sets = candidate_index_sets(ip)
    assert 1 <= len(sets) <= 4
    print(f"✓ candidate_index_sets: {len(sets)} sets for {ip}")

    # --- Test 6: isolated_disc_index_set ---
    pts5  = np.array([[0.0, 0.0], [10.0, 0.0]])
    lam5  = np.array([1.0, 1.0])
    d5    = build_discs(pts5, lam5)
    _, L2e = enumerate_regions(d5)
    s0 = isolated_disc_index_set(L2e[0], d5)
    assert L2e[0].idx in s0
    print(f"✓ isolated_disc_index_set: S={s0} for disc {L2e[0].idx}")

    # --- Test 7: S_base check — third disc containing an intersection point ---
    # Disc 0: center (0,0) r=2, Disc 1: center (2,0) r=2
    # They intersect at (1, sqrt(3)) and (1, -sqrt(3))
    # Add Disc 2: center (1,0) r=2 — should contain both intersection points
    pts6 = np.array([[0.,0.],[2.,0.],[1.,0.]])
    lam6 = np.array([4.0, 4.0, 4.0])
    d6   = build_discs(pts6, lam6)
    L1f, _ = enumerate_regions(d6)
    # Find intersection points between disc 0 and disc 1
    pts_01 = [ip for ip in L1f if {ip.i,ip.j} == {0,1}]
    for ip in pts_01:
        assert 2 in ip.S_base, \
            f"Disc 2 should be in S_base for point {ip.point}"
    print(f"✓ S_base correctly includes disc 2 for (0,1) intersection points")

    print("\nAll region enumeration tests passed.")
