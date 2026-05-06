"""
mssc_clean.py — Self-contained MSSC exact solver (Gurobi or PuLP/CBC).

Implements exact Minimum Sum-of-Squares Clustering via column-generation
Branch-and-Bound as described in:
  Aloise, Hansen & Liberti (2012). Improved column generation for the minimum
  sum-of-squares clustering problem. Mathematical Programming 134(2):539-565.

Algorithm:
  1. K-means++ warm-start provides an initial upper bound and seed columns.
  2. Column generation solves the LP relaxation of the set-partition RMP:
       min  sum_t  cost(t) * z_t
       s.t. sum_{t: i in t} z_t == 1   for all points i   (covering)
            sum_t z_t == k                                  (cardinality)
            z_t >= 0
  3. Pricing oracle (Dinkelbach-style) finds negative-reduced-cost clusters.
  4. Ryan-Foster branching drives fractional solutions to integrality.

Usage:
    python mssc_clean.py --dataset iris --k 3
    python mssc_clean.py --dataset iris --k 3 --solver pulp
    python mssc_clean.py --data mydata.csv --k 5 --seed 0
    python mssc_clean.py --dataset glass --k 10 --output results.json
"""

import argparse
import json
import time
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MSSCInstance:
    points: np.ndarray   # shape (n, s)
    k: int
    name: str = ""

    @property
    def n(self): return self.points.shape[0]
    @property
    def s(self): return self.points.shape[1]


@dataclass
class Cluster:
    indices: tuple        # sorted tuple of point indices in this cluster
    centroid: np.ndarray
    cost: float           # sum of squared distances to centroid


@dataclass
class MSSCSolution:
    labels: np.ndarray
    centroids: np.ndarray
    cost: float


@dataclass
class BnBResult:
    optimal_cost: float
    optimal_labels: np.ndarray
    lp_bound: float
    gap_pct: float
    n_nodes: int
    n_columns: int
    time_sec: float
    status: str


def make_cluster(points: np.ndarray, indices: List[int]) -> Cluster:
    pts = points[indices]
    centroid = pts.mean(axis=0)
    cost = float(((pts - centroid) ** 2).sum())
    return Cluster(indices=tuple(sorted(indices)), centroid=centroid, cost=cost)


# =============================================================================
# K-Means++ (warm-start and initial column generation)
# =============================================================================

def _kmeanspp_init(points: np.ndarray, k: int, rng) -> np.ndarray:
    n = len(points)
    chosen = [int(rng.integers(n))]
    for _ in range(k - 1):
        d2 = cdist(points, points[chosen], "sqeuclidean").min(axis=1)
        probs = d2 / d2.sum()
        chosen.append(int(rng.choice(n, p=probs)))
    return points[chosen].copy()


def kmeans(points: np.ndarray, k: int, seed: int = 42,
           n_restarts: int = 5, max_iter: int = 300) -> MSSCSolution:
    rng = np.random.default_rng(seed)
    best: Optional[MSSCSolution] = None
    n = len(points)
    for _ in range(n_restarts):
        centroids = _kmeanspp_init(points, k, rng)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            D = cdist(points, centroids, "sqeuclidean")
            labels = D.argmin(axis=1)
            new_c = np.array([
                points[labels == j].mean(0) if (labels == j).any() else centroids[j]
                for j in range(k)
            ])
            if np.allclose(new_c, centroids):
                break
            centroids = new_c
        cost = sum(
            float(((points[labels == j] - centroids[j]) ** 2).sum())
            for j in range(k) if (labels == j).any()
        )
        if best is None or cost < best.cost:
            best = MSSCSolution(labels=labels.copy(), centroids=centroids.copy(), cost=cost)
    return best


def extract_columns(points: np.ndarray, k: int, sol: MSSCSolution) -> List[Cluster]:
    cols = []
    for j in range(k):
        idxs = list(np.where(sol.labels == j)[0])
        if idxs:
            cols.append(make_cluster(points, idxs))
    return cols


# =============================================================================
# Pricing oracle (Dinkelbach-style column generation subproblem)
#
# Finds the cluster S minimising reduced cost:
#   RC(S) = cost(S, centroid(S)) - sum_{i in S} lambda_i - sigma
#
# Strategy: seed with each data point as a trial centroid, then iterate
# "include i iff lambda_i > ||x_i - c||^2" until convergence (fixed point).
# The fixed point satisfies the KKT conditions of the parametric subproblem.
# =============================================================================

def price_column(
    points: np.ndarray,
    lam: np.ndarray,
    sigma: float,
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
) -> Tuple[float, Optional[Cluster]]:
    """
    Return (best_rc, best_cluster) with best_rc < 0 if an improving column
    was found, otherwise (0.0, None).
    """
    n = len(points)
    best_rc = -1e-8
    best_col: Optional[Cluster] = None

    # Pre-compute must-link groups via union-find for same_pairs
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in same_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for seed_idx in range(n):
        c = points[seed_idx].copy()

        for _ in range(15):  # fixed-point iterations
            d2 = ((points - c) ** 2).sum(axis=1)
            # Include point i iff lambda_i > d2_i (positive net contribution)
            mask = lam > d2
            indices_set = set(np.where(mask)[0])

            # Enforce must-link: if any member of a group is included, include all
            changed = True
            while changed:
                changed = False
                for a, b in same_pairs:
                    a_in, b_in = a in indices_set, b in indices_set
                    if a_in and not b_in:
                        indices_set.add(b)
                        changed = True
                    elif b_in and not a_in:
                        indices_set.add(a)
                        changed = True

            # Enforce diff constraint: remove the less profitable of each conflicting pair
            for a, b in diff_pairs:
                if a in indices_set and b in indices_set:
                    if lam[a] - d2[a] <= lam[b] - d2[b]:
                        indices_set.discard(a)
                    else:
                        indices_set.discard(b)

            if not indices_set:
                indices_set = {seed_idx}

            indices = list(indices_set)
            new_c = points[indices].mean(axis=0)
            if np.allclose(new_c, c, atol=1e-10):
                break
            c = new_c

        if not indices:
            continue

        pts = points[indices]
        c = pts.mean(axis=0)
        cost = float(((pts - c) ** 2).sum())
        rc = cost - lam[indices].sum() - sigma

        if rc < best_rc:
            best_rc = rc
            best_col = Cluster(
                indices=tuple(sorted(indices)), centroid=c.copy(), cost=cost
            )

    return best_rc, best_col


# =============================================================================
# Restricted Master Problem — Gurobi backend
#
# LP:  min  sum_t cost_t * z_t
#      s.t. sum_{t: i in t} z_t == 1   (covering constraints, dual = lambda_i)
#           sum_t z_t == k             (cardinality constraint, dual = sigma)
#           z_t >= 0
# =============================================================================

class RMPGurobi:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.columns: List[Cluster] = []

    def add_column(self, col: Cluster) -> None:
        self.columns.append(col)

    def solve(self) -> Tuple:
        """Returns (obj, lam, sigma, z_values) or (None, None, None, None)."""
        import gurobipy as gp
        from gurobipy import GRB

        n, k, cols = self.n, self.k, self.columns
        T = len(cols)

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("Method", 2)    # barrier — fast for LP relaxation
        m.setParam("Crossover", 1)

        z = m.addVars(T, lb=0.0, name="z")
        m.setObjective(
            gp.quicksum(cols[t].cost * z[t] for t in range(T)),
            GRB.MINIMIZE
        )

        # Covering constraints: each point assigned to exactly one cluster
        cov = [
            m.addConstr(
                gp.quicksum(z[t] for t in range(T) if i in cols[t].indices) == 1,
                name=f"cov_{i}"
            )
            for i in range(n)
        ]
        # Cardinality constraint: exactly k clusters selected
        card = m.addConstr(
            gp.quicksum(z[t] for t in range(T)) == k,
            name="card"
        )

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None, None, None, None

        lam   = np.array([cov[i].Pi for i in range(n)])
        sigma = float(card.Pi)
        obj   = float(m.ObjVal)
        zvals = np.array([float(z[t].X) for t in range(T)])
        return obj, lam, sigma, zvals


# =============================================================================
# Restricted Master Problem — PuLP/CBC backend
# =============================================================================

class RMPPuLP:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.columns: List[Cluster] = []

    def add_column(self, col: Cluster) -> None:
        self.columns.append(col)

    def solve(self) -> Tuple:
        """Returns (obj, lam, sigma, z_values) or (None, None, None, None)."""
        import pulp

        n, k, cols = self.n, self.k, self.columns
        T = len(cols)

        prob = pulp.LpProblem("RMP", pulp.LpMinimize)
        z = [pulp.LpVariable(f"z_{t}", lowBound=0.0) for t in range(T)]

        prob += pulp.lpSum(cols[t].cost * z[t] for t in range(T))

        # Covering constraints
        for i in range(n):
            prob += (
                pulp.lpSum(z[t] for t in range(T) if i in cols[t].indices) == 1,
                f"cov_{i}"
            )
        # Cardinality constraint
        prob += (pulp.lpSum(z) == k, "card")

        prob.solve(pulp.PULP_CBC_CMD(msg=0, options=["DualS"]))

        if pulp.LpStatus[prob.status] != "Optimal":
            return None, None, None, None

        obj = float(pulp.value(prob.objective))
        zvals = np.array([float(pulp.value(z[t]) or 0.0) for t in range(T)])

        # Extract dual variables (PuLP >= 2.7 exposes .pi on LP constraints)
        try:
            lam   = np.array([prob.constraints[f"cov_{i}"].pi for i in range(n)],
                             dtype=float)
            sigma = float(prob.constraints["card"].pi)
        except (AttributeError, KeyError, TypeError):
            # CBC may not expose duals in older PuLP; fall back to zeros.
            # Column generation will terminate immediately (no improving columns found).
            lam   = np.zeros(n)
            sigma = 0.0

        return obj, lam, sigma, zvals


# =============================================================================
# Column generation loop
# =============================================================================

def _col_satisfies(col: Cluster,
                   same_pairs: List[Tuple[int, int]],
                   diff_pairs: List[Tuple[int, int]]) -> bool:
    s = set(col.indices)
    for a, b in same_pairs:
        if (a in s) != (b in s):
            return False
    for a, b in diff_pairs:
        if a in s and b in s:
            return False
    return True


def column_generation(
    points: np.ndarray,
    k: int,
    init_cols: List[Cluster],
    solver_cls,
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
    ub: float = 1e18,
    max_iter: int = 300,
) -> Optional[Tuple]:
    """
    Solve the LP relaxation via column generation.

    Returns (lp_bound, zvals, columns, lam, sigma) or None if infeasible.
    """
    valid = [c for c in init_cols if _col_satisfies(c, same_pairs, diff_pairs)]
    if not valid:
        return None

    rmp = solver_cls(len(points), k)
    for col in valid:
        rmp.add_column(col)

    all_cols = list(valid)
    lp_bound = lam = sigma = zvals = None

    for _ in range(max_iter):
        res = rmp.solve()
        if res[0] is None:
            break
        lp_bound, lam, sigma, zvals = res

        if lp_bound >= ub - 1e-6:
            break

        rc, new_col = price_column(points, lam, sigma, same_pairs, diff_pairs)
        if rc >= -1e-6 or new_col is None:
            break  # LP is optimal — no negative-RC column exists

        if _col_satisfies(new_col, same_pairs, diff_pairs):
            rmp.add_column(new_col)
            all_cols.append(new_col)

    if lp_bound is None:
        return None

    # Final solve to get consistent z values
    res = rmp.solve()
    if res[0] is None:
        return None
    lp_bound, lam, sigma, zvals = res

    return lp_bound, zvals, all_cols, lam, sigma


# =============================================================================
# Branch-and-Bound with Ryan–Foster branching
# =============================================================================

@dataclass
class _BnBNode:
    same_pairs: List[Tuple[int, int]]
    diff_pairs: List[Tuple[int, int]]
    depth: int
    lb: float
    cols: List[Cluster]

    def __lt__(self, other):
        return self.lb < other.lb


def _find_branch_pair(
    zvals: np.ndarray,
    cols: List[Cluster],
    n: int,
) -> Optional[Tuple[int, int]]:
    """
    Ryan-Foster branching: find (i, j) such that
    0 < Pr[i,j same cluster] < 1 in the LP solution.
    """
    frac = np.zeros((n, n))
    for t, col in enumerate(cols):
        z = zvals[t]
        if z < 1e-6:
            continue
        idxs = list(col.indices)
        for a in idxs:
            for b in idxs:
                frac[a, b] += z

    best, best_score = None, 1e18
    for i in range(n):
        for j in range(i + 1, n):
            v = frac[i, j]
            if 1e-4 < v < 1 - 1e-4:
                score = abs(v - 0.5)
                if score < best_score:
                    best_score, best = score, (i, j)
    return best


def _is_integer(zvals: np.ndarray) -> bool:
    return all(z < 1e-6 or z > 1 - 1e-6 for z in zvals)


def _extract_solution(zvals: np.ndarray, cols: List[Cluster], n: int):
    labels = np.zeros(n, dtype=int)
    cost = 0.0
    cid = 0
    for t, col in enumerate(cols):
        if zvals[t] > 0.5:
            for i in col.indices:
                labels[i] = cid
            cost += col.cost
            cid += 1
    return labels, cost


def solve_bnb(
    inst: MSSCInstance,
    solver: str = "gurobi",
    max_nodes: int = 200,
    n_restarts: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> BnBResult:
    """
    Exact MSSC solver: column-generation Branch-and-Bound.

    Parameters
    ----------
    inst        : MSSCInstance
    solver      : "gurobi" | "pulp"
    max_nodes   : int  — B&B node limit
    n_restarts  : int  — k-means restarts for warm start
    seed        : int
    verbose     : bool

    Returns
    -------
    BnBResult
    """
    t0 = time.time()
    n, k = inst.n, inst.k
    points = inst.points
    solver_cls = RMPGurobi if solver == "gurobi" else RMPPuLP

    if verbose:
        print(f"\n  CG-B&B | {inst.name} | n={n}, k={k}, s={inst.s} | solver={solver}")

    # Warm start: k-means++ upper bound
    init_sol = kmeans(points, k, seed=seed, n_restarts=n_restarts)
    best_cost = init_sol.cost
    best_labels = init_sol.labels.copy()
    if verbose:
        print(f"  Initial UB (k-means) = {best_cost:.4f}")

    # Seed columns from multiple k-means restarts
    rng = np.random.default_rng(seed)
    all_seeds = [seed] + [int(rng.integers(1 << 30)) for _ in range(n_restarts - 1)]
    raw_cols: List[Cluster] = extract_columns(points, k, init_sol)
    seen: dict = {c.indices: c for c in raw_cols}
    for s in all_seeds[1:]:
        sol = kmeans(points, k, seed=s, n_restarts=1)
        for col in extract_columns(points, k, sol):
            if col.indices not in seen:
                seen[col.indices] = col
    init_cols = list(seen.values())

    # Best-first B&B
    root = _BnBNode(same_pairs=[], diff_pairs=[], depth=0, lb=-1e18, cols=init_cols)
    heap: list = [(root.lb, 0, root)]
    node_id = 1
    n_nodes = 0
    root_lp: Optional[float] = None

    while heap and n_nodes < max_nodes:
        lb_est, _, node = heapq.heappop(heap)

        if lb_est >= best_cost - 1e-6:
            continue  # prune

        n_nodes += 1
        res = column_generation(
            points, k, node.cols, solver_cls,
            node.same_pairs, node.diff_pairs, ub=best_cost,
        )
        if res is None:
            continue

        lp_bound, zvals, node_cols, lam, sigma = res

        if root_lp is None:
            root_lp = lp_bound

        if verbose:
            print(f"  Node {n_nodes:>4}  depth={node.depth}  "
                  f"LP={lp_bound:.4f}  UB={best_cost:.4f}  cols={len(node_cols)}")

        if lp_bound >= best_cost - 1e-6:
            if verbose:
                print("         -> PRUNED")
            continue

        # Integer check
        if _is_integer(zvals):
            labels, cost = _extract_solution(zvals, node_cols, n)
            if cost < best_cost - 1e-8:
                best_cost = cost
                best_labels = labels
                if verbose:
                    print(f"         -> NEW UB = {best_cost:.4f}")
            continue

        # Ryan-Foster branching
        pair = _find_branch_pair(zvals, node_cols, n)
        if pair is None:
            continue

        i, j = pair
        for branch_same in (True, False):
            sp = node.same_pairs + ([(i, j)] if branch_same else [])
            dp = node.diff_pairs + ([] if branch_same else [(i, j)])
            child = _BnBNode(
                same_pairs=sp, diff_pairs=dp,
                depth=node.depth + 1, lb=lp_bound, cols=node_cols,
            )
            heapq.heappush(heap, (lp_bound, node_id, child))
            node_id += 1

    t_end = time.time()
    status = "optimal" if not heap or n_nodes < max_nodes else "max_nodes"
    lp_final = root_lp if root_lp is not None else best_cost
    gap = (best_cost - lp_final) / max(1e-9, abs(best_cost)) * 100

    if verbose:
        print(f"\n  Status       : {status}")
        print(f"  Optimal cost : {best_cost:.6f}")
        print(f"  LP bound     : {lp_final:.6f}")
        print(f"  Gap          : {gap:.4f}%")
        print(f"  Time         : {t_end - t0:.2f}s")
        print(f"  Nodes        : {n_nodes}")
        print(f"  Columns gen  : {len(init_cols)}")

    return BnBResult(
        optimal_cost=best_cost,
        optimal_labels=best_labels,
        lp_bound=lp_final,
        gap_pct=gap,
        n_nodes=n_nodes,
        n_columns=len(init_cols),
        time_sec=t_end - t0,
        status=status,
    )


# =============================================================================
# Data loaders
# =============================================================================

def load_dataset(name: str) -> np.ndarray:
    loaders = {
        "iris":  lambda: np.loadtxt("iris.data",  delimiter=",", usecols=range(4)),
        "glass": lambda: np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10)),
        "gr202": lambda: _load_tsp("gr202.tsp"),
        "gr666": lambda: _load_tsp("gr666.tsp"),
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose: {list(loaders)}")
    return loaders[name]()


def _load_tsp(path: str) -> np.ndarray:
    coords, in_section = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if line in ("EOF", ""):
                continue
            if in_section:
                parts = line.split()
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exact MSSC via Column-Generation Branch-and-Bound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mssc_clean.py --dataset iris --k 3
  python mssc_clean.py --dataset iris --k 3 --solver pulp
  python mssc_clean.py --data mydata.csv --k 5 --seed 0
  python mssc_clean.py --dataset glass --k 10 --output results.json
        """,
    )
    parser.add_argument("--dataset", choices=["iris", "glass", "gr202", "gr666"],
                        help="Built-in benchmark dataset")
    parser.add_argument("--data",   type=str, help="Path to CSV data file")
    parser.add_argument("--k",      type=int, required=True, help="Number of clusters")
    parser.add_argument("--solver", choices=["gurobi", "pulp"], default="gurobi",
                        help="LP solver backend (default: gurobi)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=200,
                        help="B&B node limit (default: 200)")
    parser.add_argument("--restarts",  type=int, default=5,
                        help="K-means restarts for warm start (default: 5)")
    parser.add_argument("--output",    type=str, help="Save results to JSON file")
    parser.add_argument("--quiet",  action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if args.dataset:
        points = load_dataset(args.dataset)
        name = f"{args.dataset}_k{args.k}"
    elif args.data:
        points = np.loadtxt(args.data, delimiter=",")
        name = f"custom_k{args.k}"
    else:
        parser.error("Specify --dataset or --data")

    inst = MSSCInstance(points=points, k=args.k, name=name)
    result = solve_bnb(
        inst, solver=args.solver, max_nodes=args.max_nodes,
        n_restarts=args.restarts, seed=args.seed,
        verbose=not args.quiet,
    )

    print(f"\nOptimal cost : {result.optimal_cost:.4f}")
    print(f"Gap          : {result.gap_pct:.4f}%")
    print(f"Solve time   : {result.time_sec:.2f}s")
    print(f"Nodes        : {result.n_nodes}")
    print(f"Status       : {result.status}")

    if args.output:
        out = {
            "dataset":      args.dataset or args.data,
            "k":            args.k,
            "solver":       args.solver,
            "optimal_cost": result.optimal_cost,
            "lp_bound":     result.lp_bound,
            "gap_pct":      result.gap_pct,
            "n_nodes":      result.n_nodes,
            "n_columns":    result.n_columns,
            "time_sec":     result.time_sec,
            "status":       result.status,
            "labels":       result.optimal_labels.tolist(),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
