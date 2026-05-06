"""validate_glass.py — Validate Glass dataset j-means results against paper Table 9."""
import numpy as np
from phase1_foundation import MSSCInstance
from phase8_fast_jmeans import fast_jmeans
from phase2_kmeans import kmeans

def main():
    points = np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10))

    paper = {30: 63.3284, 35: 49.2386, 40: 39.4983, 45: 32.0395, 50: 26.7675}

    print(f"\nGlass Validation — Paper Table 9 (j-means)")
    print(f"  {'k':>4}  {'paper':>10}  {'ours':>10}  {'delta%':>8}  status")
    print("  " + "-" * 50)

    for k in sorted(paper):
        inst = MSSCInstance(points, k=k, name=f"glass_k{k}")
        init = kmeans(inst, seed=42)
        sol = fast_jmeans(inst, init_solution=init, n_restarts=5, seed=42)
        delta = (sol.cost - paper[k]) / paper[k] * 100
        status = "BETTER" if delta < -0.001 else ("MATCH" if abs(delta) < 0.5 else "DIFF")
        print(f"  {k:>4}  {paper[k]:>10.4f}  {sol.cost:>10.4f}  {delta:>+8.3f}%  {status}")

if __name__ == "__main__":
    main()
