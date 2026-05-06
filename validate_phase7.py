"""
Validation: Original vs Optimised vs Combined vs Lean B&B — Four-Way
=====================================================================
Runs all four solvers on the same instances with the same seed:

  1. solve_bnb            (phase6_bnb)          — original
  2. solve_bnb_optimized  (phase6_bnb_optimized) — pruned auxiliary
  3. solve_bnb_combined   (phase7_combined_bnb)  — multi-column pricing
  4. solve_bnb_lean       (phase7_lean_bnb)      — lean child warm-start

Reports per instance:
  - Optimal cost   (correctness: all four should match)
  - B&B nodes      (search effort)
  - Columns        (RMP size at termination)
  - Wall-clock time
  - Speedup vs original

Notes on expected speedup
--------------------------
For the paper's benchmark instances (Iris, Glass, gr202, gr666), the LP
relaxation is integer-valued at the root node in virtually every case.
B&B terminates after 1 node, so:

  - Multi-column pricing (phase7_combined_bnb): overhead of a larger RMP
    cancels the benefit of fewer CG iterations on these instances.

  - Lean warm-start (phase7_lean_bnb): child nodes never fire, so the
    warm-start code path is never exercised.

Both optimisations are correct and could benefit harder instances where
the LP root is fractional and many B&B nodes are visited.

Additionally prints multi-pricing statistics for one representative
instance per dataset.

Usage
-----
    python validate_phase7.py iris
    python validate_phase7.py glass
    python validate_phase7.py gr202
    python validate_phase7.py gr666
    python validate_phase7.py all
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb
from phase6_bnb_optimized import solve_bnb_optimized
from phase7_combined_bnb import solve_bnb_combined
from phase7_lean_bnb import solve_bnb_lean


# ---------------------------------------------------------------------------
# Dataset loaders  (identical to validate_optimized.py)
# ---------------------------------------------------------------------------

def load_iris():
    return np.loadtxt("iris.data", delimiter=",", usecols=range(4))

def load_glass():
    return np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10))

def parse_tsp(path):
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


# ---------------------------------------------------------------------------
# Multi-pricing diagnostics (one CG iteration at root)
# ---------------------------------------------------------------------------

def print_multi_pricing_stats(inst, label):
    """Show how many columns the combined aux returns vs single-best."""
    from phase2_jmeans import jmeans, extract_initial_columns
    from phase3_rmp import RestrictedMasterProblem
    from phase5_algorithm2 import solve_auxiliary_general
    from phase5_algorithm2_pruned import solve_auxiliary_pruned
    from phase7_pruned_multi_aux import solve_auxiliary_combined

    jm_sol = jmeans(inst, n_restarts=5, seed=42)
    cols   = extract_initial_columns(inst, jm_sol)
    rmp    = RestrictedMasterProblem(inst)
    rmp.add_columns(cols)
    _, lam, sigma = rmp.solve()

    r_single = solve_auxiliary_pruned(inst, lam, sigma)
    r_multi  = solve_auxiliary_combined(inst, lam, sigma)

    print(f"\n  Multi-pricing stats ({label}):")
    print(f"    Cliques traversed  : {r_multi.n_cliques}")
    print(f"    Singletons         : {r_multi.n_singletons}")
    print(f"    Bound-pruned       : {r_multi.n_pruned}")
    print(f"    Dinkelbach runs    : {r_multi.n_dinkelbach}")
    print(f"    Single-best RC     : {r_single.reduced_cost:.6f}")
    print(f"    Multi  best RC     : {r_multi.best_reduced_cost:.6f}")
    print(f"    Columns returned   : {len(r_multi.clusters)}"
          f"  (vs 1 for single-best)")


# ---------------------------------------------------------------------------
# Three-way comparison runner
# ---------------------------------------------------------------------------

def compare(inst, seed=42, max_nodes=100):
    t0       = time.time()
    res_orig = solve_bnb(inst, max_nodes=max_nodes, n_jmeans_restarts=5,
                         seed=seed, verbose=False)
    t_orig   = time.time() - t0

    t0      = time.time()
    res_opt = solve_bnb_optimized(inst, max_nodes=max_nodes,
                                  n_jmeans_restarts=5,
                                  seed=seed, verbose=False)
    t_opt   = time.time() - t0

    t0       = time.time()
    res_comb = solve_bnb_combined(inst, max_nodes=max_nodes,
                                  n_jmeans_restarts=5,
                                  seed=seed, verbose=False)
    t_comb   = time.time() - t0

    t0       = time.time()
    res_lean = solve_bnb_lean(inst, max_nodes=max_nodes,
                               n_jmeans_restarts=5,
                               seed=seed, verbose=False)
    t_lean   = time.time() - t0

    ref     = res_orig.optimal_cost
    tol     = 1e-4 * max(ref, 1.0)
    ok_opt  = abs(res_opt.optimal_cost  - ref) < tol
    ok_comb = abs(res_comb.optimal_cost - ref) < tol
    ok_lean = abs(res_lean.optimal_cost - ref) < tol

    return {
        "cost_orig"  : res_orig.optimal_cost,
        "cost_opt"   : res_opt.optimal_cost,
        "cost_comb"  : res_comb.optimal_cost,
        "cost_lean"  : res_lean.optimal_cost,
        "ok_opt"     : ok_opt,
        "ok_comb"    : ok_comb,
        "ok_lean"    : ok_lean,
        "cols_orig"  : res_orig.n_columns,
        "cols_opt"   : res_opt.n_columns,
        "cols_comb"  : res_comb.n_columns,
        "cols_lean"  : res_lean.n_columns,
        "nodes_orig" : res_orig.n_nodes,
        "nodes_opt"  : res_opt.n_nodes,
        "nodes_comb" : res_comb.n_nodes,
        "nodes_lean" : res_lean.n_nodes,
        "t_orig"     : t_orig,
        "t_opt"      : t_opt,
        "t_comb"     : t_comb,
        "t_lean"     : t_lean,
        "spd_opt"    : t_orig / t_opt  if t_opt  > 0 else float("inf"),
        "spd_comb"   : t_orig / t_comb if t_comb > 0 else float("inf"),
        "spd_lean"   : t_orig / t_lean if t_lean > 0 else float("inf"),
    }


def print_header():
    print(f"\n{'k':>4}  {'cost_orig':>12}  {'ok':>4}  "
          f"{'t_orig':>7}  {'t_opt':>7}  {'t_comb':>7}  {'t_lean':>7}  "
          f"{'spd_opt':>8}  {'spd_comb':>9}  {'spd_lean':>9}")
    print("-" * 100)


def print_row(k, r):
    all_ok  = r["ok_opt"] and r["ok_comb"] and r["ok_lean"]
    ok_str  = "OK" if all_ok else "MISMATCH"
    print(f"{k:>4}  {r['cost_orig']:>12.4f}  {ok_str:>4}  "
          f"{r['t_orig']:>6.1f}s  {r['t_opt']:>6.1f}s  "
          f"{r['t_comb']:>6.1f}s  {r['t_lean']:>6.1f}s  "
          f"{r['spd_opt']:>7.2f}x  {r['spd_comb']:>8.2f}x  "
          f"{r['spd_lean']:>8.2f}x")


def print_summary(results):
    spd_opt  = [r["spd_opt"]  for r in results]
    spd_comb = [r["spd_comb"] for r in results]
    spd_lean = [r["spd_lean"] for r in results]
    mm = sum(1 for r in results
             if not (r["ok_opt"] and r["ok_comb"] and r["ok_lean"]))
    print(f"\n  Pruned-only speedup   : mean={np.mean(spd_opt):.2f}x  "
          f"max={np.max(spd_opt):.2f}x  min={np.min(spd_opt):.2f}x")
    print(f"  Combined speedup      : mean={np.mean(spd_comb):.2f}x  "
          f"max={np.max(spd_comb):.2f}x  min={np.min(spd_comb):.2f}x")
    print(f"  Lean speedup          : mean={np.mean(spd_lean):.2f}x  "
          f"max={np.max(spd_lean):.2f}x  min={np.min(spd_lean):.2f}x")
    print(f"  Cost mismatches       : {mm}  (0 = all correct)")


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

def run_iris():
    print("\n" + "="*80)
    print("  Fisher's Iris  n=150  s=4  (Paper Table 8)")
    print("="*80)
    points = load_iris()
    print_multi_pricing_stats(MSSCInstance(points, 5, name="iris_k5"), "Iris k=5")
    results = []
    print_header()
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        inst = MSSCInstance(points, k, name=f"iris_k{k}")
        r    = compare(inst)
        print_row(k, r)
        results.append(r)
    print_summary(results)


def run_glass():
    print("\n" + "="*80)
    print("  Glass Identification  n=214  s=9  (Paper Table 9)")
    print("="*80)
    points = load_glass()
    print_multi_pricing_stats(MSSCInstance(points, 30, name="glass_k30"), "Glass k=30")
    results = []
    print_header()
    for k in [30, 35, 40, 45, 50]:
        inst = MSSCInstance(points, k, name=f"glass_k{k}")
        r    = compare(inst)
        print_row(k, r)
        results.append(r)
    print_summary(results)


def run_gr202():
    print("\n" + "="*80)
    print("  Grotschel 202-city  n=202  s=2  (Paper Table 3)")
    print("="*80)
    points  = parse_tsp("gr202.tsp")
    results = []
    print_header()
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
        inst = MSSCInstance(points, k, name=f"gr202_k{k}")
        r    = compare(inst)
        print_row(k, r)
        results.append(r)
    print_summary(results)


def run_gr666():
    print("\n" + "="*80)
    print("  Grotschel 666-city  n=666  s=2  (Paper Table 4)")
    print("="*80)
    points  = parse_tsp("gr666.tsp")
    results = []
    print_header()
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        inst = MSSCInstance(points, k, name=f"gr666_k{k}")
        r    = compare(inst)
        print_row(k, r)
        results.append(r)
    print_summary(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    "iris"  : run_iris,
    "glass" : run_glass,
    "gr202" : run_gr202,
    "gr666" : run_gr666,
}


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python validate_phase7.py "
              "[iris|glass|gr202|gr666|all]")
        return
    if "all" in args:
        args = list(RUNNERS.keys())
    for name in args:
        name = name.lower()
        if name not in RUNNERS:
            print(f"Unknown dataset: {name}")
            continue
        RUNNERS[name]()


if __name__ == "__main__":
    main()
