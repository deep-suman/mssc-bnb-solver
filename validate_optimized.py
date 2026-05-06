"""validate_optimized.py — Compare optimized B&B (Phase 6) vs original on key instances."""
import time
import numpy as np
from phase1_foundation import MSSCInstance


def load_iris():
    return np.loadtxt("iris.data", delimiter=",", usecols=range(4))


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


def compare(points, k, name, seed=42, max_nodes=200):
    from phase6_bnb import solve_bnb
    from phase6_bnb_optimized import solve_bnb_optimized

    inst = MSSCInstance(points, k=k, name=name)

    t0 = time.time()
    orig = solve_bnb(inst, max_nodes=max_nodes, n_jmeans_restarts=5, seed=seed, verbose=False)
    t_orig = time.time() - t0

    t0 = time.time()
    opt = solve_bnb_optimized(inst, max_nodes=max_nodes, n_jmeans_restarts=5, seed=seed, verbose=False)
    t_opt = time.time() - t0

    match = abs(orig.optimal_cost - opt.optimal_cost) / max(1e-9, abs(orig.optimal_cost)) < 1e-5
    speedup = t_orig / max(t_opt, 1e-6)

    return {
        "k": k,
        "orig_cost": orig.optimal_cost,
        "opt_cost": opt.optimal_cost,
        "t_orig": t_orig,
        "t_opt": t_opt,
        "speedup": speedup,
        "match": match,
    }


def main():
    print(f"\n  {'Dataset':>8}  {'k':>4}  {'orig_cost':>14}  {'opt_cost':>14}  {'speedup':>8}  match")
    print("  " + "-" * 66)

    iris = load_iris()
    for k in [2, 5, 10]:
        r = compare(iris, k, f"iris_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        print(f"  {'Iris':>8}  {k:>4}  {r['orig_cost']:>14.4f}  {r['opt_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")

    gr202 = load_tsp("gr202.tsp")
    for k in [5, 15, 30]:
        r = compare(gr202, k, f"gr202_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        print(f"  {'gr202':>8}  {k:>4}  {r['orig_cost']:>14.4f}  {r['opt_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")

    gr666 = load_tsp("gr666.tsp")
    for k in [5, 10]:
        r = compare(gr666, k, f"gr666_k{k}")
        flag = "OK" if r["match"] else "MISMATCH"
        print(f"  {'gr666':>8}  {k:>4}  {r['orig_cost']:>14.4f}  {r['opt_cost']:>14.4f}  "
              f"{r['speedup']:>7.2f}x  {flag}")

    print()


if __name__ == "__main__":
    main()
