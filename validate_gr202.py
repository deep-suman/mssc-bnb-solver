"""
Validation script: Grötschel 202-city dataset — Table 3 of the paper.
n=202, s=2. Uses Phase 4 (2D auxiliary) since s=2.

Parse gr202.tsp NODE_COORD_SECTION, treat x,y as 2D Euclidean points.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb


# ---------- parser --------------------------------------------------------

def parse_gr202(path):
    coords = []
    in_section = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if line in ("EOF", ""):
                if in_section:
                    break
                continue
            if in_section:
                parts = line.split()
                # parts: [id, x, y]
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)


# ---------- Table 3 reference values --------------------------------------

TABLE3 = {
    2:  23437.4,
    3:  15327.4,
    4:  11455.6,
    5:   8894.90,
    6:   6764.88,
    7:   5817.57,
    8:   5006.10,
    9:   4376.19,
    10:  3792.49,
    15:  2320.08,
    20:  1523.51,
    25:  1085.56,
    30:   799.311,
}


# ---------- main ----------------------------------------------------------

def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr202.tsp")
    points = parse_gr202(path)
    print(f"Loaded {len(points)} cities from gr202.tsp")
    assert len(points) == 202, f"Expected 202 points, got {len(points)}"

    # Run k values from small to large; stop early if one goes wrong
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

    print(f"\n{'k':>4}  {'fopt_paper':>14}  {'our_cost':>14}  {'gap%':>8}  "
          f"{'nodes':>6}  {'cols':>6}  {'time':>8}  status")
    print("-" * 80)

    for k in k_values:
        inst = MSSCInstance(points, k, name=f"gr202_k{k}")
        t0 = time.time()
        res = solve_bnb(inst, max_nodes=100, n_jmeans_restarts=5, seed=42, verbose=False)
        elapsed = time.time() - t0

        paper = TABLE3[k]
        gap = (res.optimal_cost - paper) / paper * 100

        print(f"{k:>4}  {paper:>14.4f}  {res.optimal_cost:>14.4f}  "
              f"{gap:>+8.3f}%  {res.n_nodes:>6}  {res.n_columns:>6}  "
              f"{elapsed:>7.1f}s  {res.status}")


if __name__ == "__main__":
    main()
