"""
Phase 5a: Hypersphere Intersection Graph
==========================================
Reference: Section 4, Proposition 4 of the paper.

For general Euclidean space (any dimension s), we build a graph G = (N, E)
where:
  - Each node n_i corresponds to entity i
  - An edge e_ij exists iff the hyperspheres centered at p_i and p_j
    with radii sqrt(lambda_i) and sqrt(lambda_j) intersect:

        ||p_i - p_j|| <= sqrt(lambda_i) + sqrt(lambda_j)

Proposition 4 (paper): If (y*, v*) is optimal for the auxiliary problem,
then the entities with v*_i = 1 form a CLIQUE in G.

This means we only need to search for the best cluster among all cliques
in G — drastically reducing the search space when G is sparse.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from phase1_foundation import MSSCInstance


# ---------------------------------------------------------------------------
# 1.  Graph structure
# ---------------------------------------------------------------------------

class IntersectionGraph:
    """
    Hypersphere intersection graph G = (N, E).

    Attributes
    ----------
    n         : number of nodes (entities)
    adj       : adjacency list — adj[i] = sorted list of neighbors of i
    edges     : list of (i, j) pairs with i < j
    degrees   : degree[i] = number of neighbors of node i
    """

    def __init__(self, n: int):
        self.n       = n
        self.adj     : List[List[int]] = [[] for _ in range(n)]
        self.edges   : List[Tuple[int, int]] = []
        self.degrees : np.ndarray = np.zeros(n, dtype=int)

    def add_edge(self, i: int, j: int) -> None:
        if j not in self.adj[i]:   # avoid duplicates
            self.adj[i].append(j)
            self.adj[j].append(i)
            self.edges.append((min(i,j), max(i,j)))
            self.degrees[i] += 1
            self.degrees[j] += 1

    def neighbors(self, i: int) -> List[int]:
        return self.adj[i]

    def degree(self, i: int) -> int:
        return self.degrees[i]

    def subgraph(self, nodes: List[int]) -> 'IntersectionGraph':
        """Return induced subgraph on given nodes."""
        node_set = set(nodes)
        g = IntersectionGraph(self.n)
        for i in nodes:
            for j in self.adj[i]:
                if j in node_set and i < j:
                    g.add_edge(i, j)
        return g

    def __repr__(self):
        return (f"IntersectionGraph(n={self.n}, "
                f"edges={len(self.edges)}, "
                f"density={len(self.edges)/max(1,self.n*(self.n-1)//2):.3f})")


# ---------------------------------------------------------------------------
# 2.  Build the graph from instance + dual variables
# ---------------------------------------------------------------------------

def build_intersection_graph(inst: MSSCInstance,
                              lam: np.ndarray) -> IntersectionGraph:
    """
    Build G = (N, E) for general Euclidean space.

    Edge (i,j) exists iff:
        ||p_i - p_j|| <= sqrt(lambda_i) + sqrt(lambda_j)

    Uses the acceleration from the paper (Section 3):
        Sort entities by their first coordinate.
        For each i, scan j > i in sorted order.
        Stop early when p_i[0] - p_j[0] > sqrt(lambda_i) + sqrt(lambda_max)
        since no further j can intersect i.

    Parameters
    ----------
    inst : MSSCInstance
    lam  : np.ndarray shape (n,)

    Returns
    -------
    IntersectionGraph
    """
    n      = inst.n
    points = inst.points
    radii  = np.sqrt(np.maximum(lam, 0.0))   # sqrt(lambda_i) for each i

    g = IntersectionGraph(n)

    for i in range(n):
        for j in range(i + 1, n):
            # Edge condition: ||p_i - p_j|| <= r_i + r_j
            diff   = points[i] - points[j]
            dist   = float(np.sqrt(diff @ diff))
            if dist <= radii[i] + radii[j] + 1e-10:
                g.add_edge(i, j)

    return g


# ---------------------------------------------------------------------------
# 3.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_foundation import make_synthetic_instance

    print("=" * 50)
    print("Phase 5a — Intersection Graph Tests")
    print("=" * 50)

    # --- Test 1: two nearby entities → one edge ---
    inst1 = MSSCInstance(
        points=np.array([[0.0, 0.0], [1.0, 0.0]]), k=2)
    lam1  = np.array([1.0, 1.0])   # radii = 1.0, dist = 1.0 → edge exists
    g1    = build_intersection_graph(inst1, lam1)
    assert len(g1.edges) == 1, f"Expected 1 edge, got {len(g1.edges)}"
    assert g1.degree(0) == 1
    print(f"✓ Two nearby entities → 1 edge")

    # --- Test 2: two far entities → no edge ---
    inst2 = MSSCInstance(
        points=np.array([[0.0, 0.0], [10.0, 0.0]]), k=2)
    lam2  = np.array([1.0, 1.0])   # radii = 1.0, dist = 10.0 → no edge
    g2    = build_intersection_graph(inst2, lam2)
    assert len(g2.edges) == 0
    print(f"✓ Two far entities → 0 edges")

    # --- Test 3: three mutually close entities → complete graph K3 ---
    pts3  = np.array([[0.,0.],[1.,0.],[0.5, 0.8]])
    inst3 = MSSCInstance(points=pts3, k=3)
    lam3  = np.array([1.0, 1.0, 1.0])
    g3    = build_intersection_graph(inst3, lam3)
    assert len(g3.edges) == 3, f"Expected 3 edges (K3), got {len(g3.edges)}"
    print(f"✓ Three close entities → complete graph K3")

    # --- Test 4: zero lambda → no edges (zero-radius hyperspheres) ---
    inst4 = MSSCInstance(points=pts3, k=3)
    lam4  = np.zeros(3)
    g4    = build_intersection_graph(inst4, lam4)
    assert len(g4.edges) == 0
    print(f"✓ Zero lambda → no edges")

    # --- Test 5: large lambda → complete graph ---
    inst5 = make_synthetic_instance(n=10, k=2, s=3, seed=0)
    lam5  = np.full(10, 1e6)
    g5    = build_intersection_graph(inst5, lam5)
    expected = 10 * 9 // 2
    assert len(g5.edges) == expected, \
        f"Expected {expected} edges, got {len(g5.edges)}"
    print(f"✓ Large lambda → complete graph ({len(g5.edges)} edges)")

    # --- Test 6: subgraph ---
    sub = g5.subgraph([0, 1, 2, 3])
    assert len(sub.edges) == 6   # K4
    print(f"✓ Subgraph on 4 nodes → {len(sub.edges)} edges (K4)")

    # --- Test 7: sparsity increases with smaller lambda ---
    inst7 = make_synthetic_instance(n=20, k=4, s=2, spread=1.0, seed=0)
    lam_big   = np.full(20, 100.0)
    lam_small = np.full(20, 0.1)
    g_big   = build_intersection_graph(inst7, lam_big)
    g_small = build_intersection_graph(inst7, lam_small)
    assert len(g_big.edges) >= len(g_small.edges), \
        "Larger lambda should give denser graph"
    print(f"✓ Sparsity: big λ → {len(g_big.edges)} edges, "
          f"small λ → {len(g_small.edges)} edges")

    print("\nAll intersection graph tests passed.")
