"""
validate_phase8.py — Compare Phase 8 fast B&B vs original B&B on benchmark datasets.

Usage:
    python validate_phase8.py iris
    python validate_phase8.py glass
    python validate_phase8.py gr202
    python validate_phase8.py gr666
    python validate_phase8.py all
"""

import sys
import time
import numpy as np

from phase1_foundation import MSSCInstance


def load_iris():
    points = np.loadtxt("iris.data", delimiter=",", usecols=range(4))
    return points


def load_glass():
    points = np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10))
    return points


def load_tsp(path):
    coords = []
    in_section = False
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


def run_bnb_comparison(points, k, name, seed=42, max_nodes=200):
    from phase6_bnb import solve_bnb
    from phase8_bnb import solve_bnb_fast

    inst = MSSCInstance(points, k=k, name=name)

    t0 = time.time()
    orig = solve_bnb(inst, max_nodes=max_nodes, n_jmeans_restarts=5, seed=seed, verbose=False)
    t_orig = time.time() - t0

    t0 = time.time()
    fast = solve_bnb_fast(inst, max_nodes=max_nodes, n_jmeans_restarts=5, seed=seed, verbose=False)
    t_fast = time.time() - t0

    match = abs(orig.optimal_cost - fast.optimal_cost) / max(1e-9, abs(orig.optimal_cost)) < 1e-5
    speedup = t_orig / max(t_fast, 1e-6)

    return {
        "k": k,
        "orig_cost": orig.optimal_cost,
        "fast_cost": fast.optimal_cost,
        "t_orig": t_orig,
        "t_fast": t_fast,
        "speedup": speedup,
        "match": match,
        "orig_status": orig.status,
        "fast_status": fast.status,
    }


def run_jmeans_comparison(points, k, name, seed=42, n_restarts=10):
    """For Glass: compare j-means quality only (B&B too slow)."""
    from phase2_jmeans import jmeans
    from phase8_fast_jmeans import fast_jmeans
    from phase2_kmeans import kmeans

    inst = MSSCInstance(points, k=k, name=name)
    init_sol = kmeans(inst, seed=seed)

    t0 = time.time()
    orig_sol = jmeans(inst, init_solution=init_sol, n_restarts=n_restarts, seed=seed)
    t_orig = time.time() - t0

    t0 = time.time()
    fast_sol = fast_jmeans(inst, init_solution=init_sol, n_restarts=n_restarts, seed=seed)
    t_fast = time.time() - t0

    speedup = t_orig / max(t_fast, 1e-6)
    delta = (fast_sol.cost - orig_sol.cost) / max(1e-9, abs(orig_sol.cost)) * 100

    return {
        "k": k,
        "orig_cost": orig_sol.cost,
        "fast_cost": fast_sol.cost,
        "t_orig": t_orig,
        "t_fast": t_fast,
        "speedup": speedup,
        "delta_pct": delta,
    }


def validate_iris():
    print("\nIris B&B — Phase 8 vs Original")
    print(f"  {'k':>4}  {'orig_cost':>14}  {'fast_cost':>14}  {'speedup':>8}  match")
    print("  " + "-" * 60)
    points = load_iris()
    all_match = True
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r = run_bnb_comparison(points, k, f"iris_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        if not r["match"]:
            all_match = False
        print(f"  {k:>4}  {r['orig_cost']:>14.4f}  {r['fast_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")
    print(f"\n  Result: {'ALL CORRECT' if all_match else 'FAILURES DETECTED'}")


def validate_glass():
    print("\nGlass J-Means — Phase 8 vs Original (10 restarts per k)")
    print(f"  {'k':>4}  {'orig_cost':>12}  {'fast_cost':>12}  {'delta%':>8}  {'speedup':>8}")
    print("  " + "-" * 60)
    points = load_glass()
    for k in [30, 35, 40, 45, 50]:
        r = run_jmeans_comparison(points, k, f"glass_k{k}")
        print(f"  {k:>4}  {r['orig_cost']:>12.4f}  {r['fast_cost']:>12.4f}  "
              f"{r['delta_pct']:>+8.3f}%  {r['speedup']:>7.2f}x")


def validate_gr202():
    print("\ngr202 B&B — Phase 8 vs Original")
    print(f"  {'k':>4}  {'orig_cost':>14}  {'fast_cost':>14}  {'speedup':>8}  match")
    print("  " + "-" * 60)
    points = load_tsp("gr202.tsp")
    all_match = True
    for k in [2, 5, 10, 15, 20, 25, 30]:
        r = run_bnb_comparison(points, k, f"gr202_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        if not r["match"]:
            all_match = False
        print(f"  {k:>4}  {r['orig_cost']:>14.4f}  {r['fast_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")
    print(f"\n  Result: {'ALL CORRECT' if all_match else 'FAILURES DETECTED'}")


def validate_gr666():
    print("\ngr666 B&B — Phase 8 vs Original")
    print(f"  {'k':>4}  {'orig_cost':>14}  {'fast_cost':>14}  {'speedup':>8}  match")
    print("  " + "-" * 60)
    points = load_tsp("gr666.tsp")
    all_match = True
    for k in [2, 5, 10, 20, 30, 40, 50]:
        r = run_bnb_comparison(points, k, f"gr666_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        if not r["match"]:
            all_match = False
        print(f"  {k:>4}  {r['orig_cost']:>14.4f}  {r['fast_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")
    print(f"\n  Result: {'ALL CORRECT' if all_match else 'FAILURES DETECTED'}")


if __name__ == "__main__":
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else "iris"
    if dataset == "iris":
        validate_iris()
    elif dataset == "glass":
        validate_glass()
    elif dataset == "gr202":
        validate_gr202()
    elif dataset == "gr666":
        validate_gr666()
    elif dataset == "all":
        validate_iris()
        validate_gr202()
        validate_gr666()
        validate_glass()
    else:
        print(f"Unknown dataset: {dataset}. Use iris/glass/gr202/gr666/all")
