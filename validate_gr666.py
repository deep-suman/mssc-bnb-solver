"""
Validation script: Grötschel 666-city dataset — Table 4 of the paper.
n=666, s=2. Uses Phase 4 (2D auxiliary).

Note: Paper's Table 4 only reports accpm-vns-a1 and accpm-a1 times,
not our algorithm. fopt values are the exact optimal costs to match.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb


def parse_tsp_geo(path):
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
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)


# Table 4 reference values
TABLE4 = {
    2:  1754012.0,
    3:   772707.0,
    4:   613995.0,
    5:   485088.0,
    6:   382676.0,
    7:   323283.0,
    8:   285925.0,
    9:   250989.0,
    10:  224183.0,
    20:  106276.0,
    50:   35179.5,
}


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr666.tsp")
    points = parse_tsp_geo(path)
    print(f"Loaded {len(points)} cities from gr666.tsp")
    assert len(points) == 666

    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

    print(f"\n{'k':>4}  {'fopt_paper':>14}  {'our_cost':>14}  {'gap%':>8}  "
          f"{'nodes':>6}  {'cols':>6}  {'time':>8}  status")
    print("-" * 80)

    for k in k_values:
        inst = MSSCInstance(points, k, name=f"gr666_k{k}")
        t0 = time.time()
        res = solve_bnb(inst, max_nodes=100, n_jmeans_restarts=5, seed=42, verbose=False)
        elapsed = time.time() - t0

        paper = TABLE4[k]
        gap = (res.optimal_cost - paper) / paper * 100

        print(f"{k:>4}  {paper:>14.1f}  {res.optimal_cost:>14.4f}  "
              f"{gap:>+8.3f}%  {res.n_nodes:>6}  {res.n_columns:>6}  "
              f"{elapsed:>7.1f}s  {res.status}")


if __name__ == "__main__":
    main()
