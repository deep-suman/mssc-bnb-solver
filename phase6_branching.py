"""
Phase 6a: Ryan-Foster Branching Rule
======================================
Reference: Section 2 (branching), page 200 of the paper.

After solving the LP relaxation, if the solution is fractional we need
to branch. The Ryan-Foster rule (Ryan & Foster, 1981) is:

  Find two entities i1, i2 such that there exist two columns t1, t2
  with FRACTIONAL z values where:
    a_{i1,t1} = a_{i2,t1} = 1   (i1 and i2 are TOGETHER in t1)
    a_{i1,t2} = 1, a_{i2,t2} = 0  (i1 is in t2, i2 is NOT in t2)

  Then branch into two subproblems:
    Branch SAME : force i1 and i2 into the SAME cluster
                  constraint: v_{i1} = v_{i2}   (type I1 in paper)
    Branch DIFF : force i1 and i2 into DIFFERENT clusters
                  constraint: v_{i1} + v_{i2} <= 1  (type I2 in paper)

Why Ryan-Foster?
  Standard branching on z_t (set z_t=0 or z_t=1) destroys the column
  generation structure. Ryan-Foster branches on ENTITY PAIRS instead,
  which naturally translates into constraints on the auxiliary problem
  (equations 8 and 11 in the paper).

This file implements:
  1. BranchNode — holds branching constraints for one B&B node
  2. find_branching_pair — detects the Ryan-Foster pair from LP solution
  3. is_integer_solution — checks if current LP solution is integer
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from phase1_foundation import Cluster


# ---------------------------------------------------------------------------
# 1.  Branch node
# ---------------------------------------------------------------------------

@dataclass
class BranchNode:
    """
    One node in the branch-and-bound tree.

    Attributes
    ----------
    same_pairs : list of (i, j) — entities forced into the SAME cluster
    diff_pairs : list of (i, j) — entities forced into DIFFERENT clusters
    depth      : depth in the B&B tree
    lb         : LP lower bound at this node (filled after solving)
    """
    same_pairs : List[Tuple[int,int]] = field(default_factory=list)
    diff_pairs : List[Tuple[int,int]] = field(default_factory=list)
    depth      : int   = 0
    lb         : float = -np.inf

    def is_root(self) -> bool:
        return len(self.same_pairs) == 0 and len(self.diff_pairs) == 0

    def branch_same(self, i: int, j: int) -> 'BranchNode':
        """Create child node forcing i and j into the SAME cluster."""
        return BranchNode(
            same_pairs = self.same_pairs + [(i, j)],
            diff_pairs = self.diff_pairs.copy(),
            depth      = self.depth + 1
        )

    def branch_diff(self, i: int, j: int) -> 'BranchNode':
        """Create child node forcing i and j into DIFFERENT clusters."""
        return BranchNode(
            same_pairs = self.same_pairs.copy(),
            diff_pairs = self.diff_pairs + [(i, j)],
            depth      = self.depth + 1
        )

    def __repr__(self):
        return (f"BranchNode(depth={self.depth}, "
                f"same={self.same_pairs}, "
                f"diff={self.diff_pairs}, "
                f"lb={self.lb:.4f})")


# ---------------------------------------------------------------------------
# 2.  Check if LP solution is integer
# ---------------------------------------------------------------------------

def is_integer_solution(z_values: np.ndarray,
                         tol: float = 1e-4) -> bool:
    """
    Return True if all z variables are within tol of 0 or 1.
    """
    for z in z_values:
        if tol < z < 1.0 - tol:
            return False
    return True


# ---------------------------------------------------------------------------
# 3.  Ryan-Foster branching pair detection
# ---------------------------------------------------------------------------

def find_branching_pair(columns   : List[Cluster],
                         z_values  : np.ndarray,
                         tol       : float = 1e-4
                         ) -> Optional[Tuple[int, int]]:
    """
    Find a Ryan-Foster branching pair (i1, i2).

    Algorithm:
      For each pair of fractional columns (t1, t2) with z_t1, z_t2 > tol:
        Find entities in t1 but not t2, and vice versa.
        If such entities exist, return the pair (i1, i2) where:
          i1 is in BOTH t1 and t2
          i2 is in t1 but NOT t2

    We prefer pairs where both columns have z values close to 0.5
    (most fractional) as this gives the most balanced branching.

    Returns
    -------
    (i1, i2) or None if solution is integer
    """
    # Collect fractional columns
    frac_cols = [
        (t, z_values[t], set(columns[t].indices))
        for t in range(len(columns))
        if tol < z_values[t] < 1.0 - tol
    ]

    if len(frac_cols) < 2:
        return None

    best_pair   = None
    best_score  = -1.0   # prefer pairs with z values closest to 0.5

    for a in range(len(frac_cols)):
        t1, z1, idx1 = frac_cols[a]
        for b in range(a + 1, len(frac_cols)):
            t2, z2, idx2 = frac_cols[b]

            # Entities in t1 but not t2
            only_in_t1 = idx1 - idx2
            # Entities in both
            in_both    = idx1 & idx2

            if len(only_in_t1) == 0 or len(in_both) == 0:
                continue

            # Found a valid Ryan-Foster pair
            i2 = min(only_in_t1)   # in t1, not in t2
            i1 = min(in_both)      # in both t1 and t2

            # Score: how fractional are the two columns?
            score = 1.0 - abs(z1 - 0.5) - abs(z2 - 0.5)
            if score > best_score:
                best_score = score
                best_pair  = (i1, i2)

    return best_pair


# ---------------------------------------------------------------------------
# 4.  Filter columns that violate branching constraints
# ---------------------------------------------------------------------------

def column_satisfies_constraints(cluster    : Cluster,
                                  node       : BranchNode) -> bool:
    """
    Check if a cluster (column) satisfies all branching constraints
    of a B&B node.

    SAME constraint (i,j): both i and j must be in the cluster,
                           or neither.
    DIFF constraint (i,j): i and j cannot both be in the cluster.
    """
    idx = set(cluster.indices)

    for i, j in node.same_pairs:
        i_in = i in idx
        j_in = j in idx
        if i_in != j_in:
            return False   # one is in, one is out → violates SAME

    for i, j in node.diff_pairs:
        if i in idx and j in idx:
            return False   # both in → violates DIFF

    return True


# ---------------------------------------------------------------------------
# 5.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance, make_cluster
    import numpy as np

    print("=" * 50)
    print("Phase 6a — Branching Tests")
    print("=" * 50)

    inst = make_synthetic_instance(n=10, k=3, s=2, seed=0)

    # --- Test 1: BranchNode creation and children ---
    root = BranchNode()
    assert root.is_root()
    child_same = root.branch_same(0, 1)
    child_diff = root.branch_diff(0, 1)
    assert child_same.same_pairs == [(0, 1)]
    assert child_diff.diff_pairs == [(0, 1)]
    assert child_same.depth == 1
    print("✓ BranchNode children created correctly")

    # --- Test 2: is_integer_solution ---
    assert is_integer_solution(np.array([1.0, 0.0, 1.0]))
    assert not is_integer_solution(np.array([1.0, 0.5, 0.0]))
    assert not is_integer_solution(np.array([0.3, 0.7]))
    print("✓ is_integer_solution correct")

    # --- Test 3: find_branching_pair on fractional solution ---
    # Two fractional columns: t1={0,1,2}, t2={1,2,3}
    # z_t1=0.5, z_t2=0.5
    # i1=1 (in both), i2=0 (in t1 not t2) → pair (1,0) or (0,1)
    col1 = make_cluster(inst, [0, 1, 2])
    col2 = make_cluster(inst, [1, 2, 3])
    col3 = make_cluster(inst, [4, 5, 6])
    cols = [col1, col2, col3]
    z    = np.array([0.5, 0.5, 0.0])
    pair = find_branching_pair(cols, z)
    assert pair is not None, "Should find a branching pair"
    assert set(pair) <= {0, 1, 2, 3}, f"Pair {pair} should be from cols"
    print(f"✓ Branching pair found: {pair}")

    # --- Test 4: no pair when solution is integer ---
    z_int = np.array([1.0, 0.0, 1.0])
    pair2 = find_branching_pair(cols, z_int)
    assert pair2 is None
    print("✓ No branching pair for integer solution")

    # --- Test 5: column_satisfies_constraints ---
    node = BranchNode(same_pairs=[(0,1)], diff_pairs=[(2,3)])

    # Cluster {0,1,4}: 0 and 1 together → satisfies SAME(0,1)
    #                  no 2 and 3 together → satisfies DIFF(2,3)
    cl_ok = make_cluster(inst, [0, 1, 4])
    assert column_satisfies_constraints(cl_ok, node)
    print("✓ Valid column passes constraints")

    # Cluster {0,4}: 0 in, 1 not in → violates SAME(0,1)
    cl_bad_same = make_cluster(inst, [0, 4])
    assert not column_satisfies_constraints(cl_bad_same, node)
    print("✓ SAME violation correctly detected")

    # Cluster {2,3,4}: both 2 and 3 in → violates DIFF(2,3)
    cl_bad_diff = make_cluster(inst, [2, 3, 4])
    assert not column_satisfies_constraints(cl_bad_diff, node)
    print("✓ DIFF violation correctly detected")

    # --- Test 6: nested branching constraints ---
    child = node.branch_same(4, 5)
    assert child.same_pairs == [(0,1),(4,5)]
    assert child.diff_pairs == [(2,3)]
    assert child.depth == 1
    print("✓ Nested branching constraints correct")

    print("\nAll branching tests passed.")
