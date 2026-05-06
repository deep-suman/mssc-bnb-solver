"""validate_iris.py — Validate Iris dataset results against paper Table 8."""
import numpy as np
from phase1_foundation import MSSCInstance
from phase8_bnb import solve_bnb_fast

def main():
    points = np.loadtxt("iris.data", delimiter=",", usecols=range(4))

    paper = {2: 152.3687, 3: 94.3061, 4: 57.2284, 5: 46.5356,
             6: 39.0397, 7: 34.8811, 8: 30.3134, 9: 27.9798, 10: 25.8134}

    print(f"\nIris Validation — Paper Table 8")
    print(f"  {'k':>4}  {'paper':>12}  {'ours':>12}  {'gap%':>8}  status")
    print("  " + "-" * 55)

    for k in sorted(paper):
        inst = MSSCInstance(points, k=k, name=f"iris_k{k}")
        result = solve_bnb_fast(inst, max_nodes=200, n_jmeans_restarts=5, seed=42, verbose=False)
        gap = abs(result.optimal_cost - paper[k]) / paper[k] * 100
        status = "MATCH" if gap < 0.01 else "DIFF"
        print(f"  {k:>4}  {paper[k]:>12.4f}  {result.optimal_cost:>12.4f}  "
              f"{gap:>8.4f}%  {status}")

if __name__ == "__main__":
    main()
