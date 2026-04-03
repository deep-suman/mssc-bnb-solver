"""
Validation script: Body measurements dataset — Table 10 of the paper.
n=507, s=5. Uses Phase 5 (general Euclidean auxiliary).

Data: body.dat.txt — 507 rows, 25 space-separated columns.
Columns 0-4 are the first 5 skeletal diameter measurements:
  0: Biacromial diameter
  1: Biiliac diameter
  2: Bitrochanteric diameter
  3: Chest depth
  4: Chest diameter
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb


# ---------- Table 10 reference values -------------------------------------

TABLE10 = {
    30: 19529.9,
    40: 16231.8,
    50: 13954.7,
    60: 12182.6,
    70: 10786.9,
    80:  9648.73,
}


# ---------- main ----------------------------------------------------------

def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body.dat.txt")
    data = np.loadtxt(path)
    assert data.shape == (507, 25), f"Expected (507,25), got {data.shape}"

    # Use first 5 columns: the 5 skeletal diameter measurements
    points = data[:, [9, 10, 11, 22, 23]]
    print(f"Loaded body.dat.txt: {points.shape[0]} observations, using cols 9,10,11,22,23 (shoulder_g, chest_g, waist_g, weight, height)")
    print(f"  col means: {points.mean(axis=0).round(2)}")

    k_values = [30, 40, 50, 60, 70, 80]

    print(f"\n{'k':>4}  {'fopt_paper':>14}  {'our_cost':>14}  {'gap%':>8}  "
          f"{'nodes':>6}  {'cols':>6}  {'time':>9}  status")
    print("-" * 82)

    for k in k_values:
        inst = MSSCInstance(points, k, name=f"body_k{k}")
        t0 = time.time()
        res = solve_bnb(inst, max_nodes=100, n_jmeans_restarts=5, seed=42, verbose=False)
        elapsed = time.time() - t0

        paper = TABLE10[k]
        gap = (res.optimal_cost - paper) / paper * 100

        print(f"{k:>4}  {paper:>14.4f}  {res.optimal_cost:>14.4f}  "
              f"{gap:>+8.3f}%  {res.n_nodes:>6}  {res.n_columns:>6}  "
              f"{elapsed:>8.1f}s  {res.status}")


if __name__ == "__main__":
    main()
