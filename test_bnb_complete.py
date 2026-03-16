import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb
from phase2_jmeans import jmeans
from phase2_kmeans import kmeans

# Overlapping Gaussian clusters — j-means often suboptimal
np.random.seed(123)
k, n_per = 4, 15
centers  = np.array([[0,0],[6,0],[3,5],[9,5]], dtype=float)
points   = np.vstack([
    centers[i] + np.random.normal(0, 1.5, (n_per, 2))
    for i in range(k)
])
inst = MSSCInstance(points=points, k=k, name="overlapping_k4")

# Run j-means many times to get best known UB
best_ub = np.inf
for seed in range(20):
    sol = jmeans(inst, n_restarts=3, seed=seed)
    if sol.cost < best_ub:
        best_ub = sol.cost
print(f"Best j-means UB (20 restarts): {best_ub:.4f}")

# Run B&B
res = solve_bnb(inst, verbose=True, max_nodes=100,
                n_jmeans_restarts=10, seed=42)

print(f"\nResult : {res}")
print(f"B&B UB : {res.optimal_cost:.4f}")
print(f"Best j-means: {best_ub:.4f}")
print(f"B&B improved j-means: {res.optimal_cost < best_ub - 1e-4}")
