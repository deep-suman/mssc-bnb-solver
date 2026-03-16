import numpy as np
import inspect
from phase1_foundation import MSSCInstance, Cluster
from phase3_rmp import RestrictedMasterProblem
from phase2_jmeans import jmeans

np.random.seed(123)
k, n_per = 4, 15
centers  = np.array([[0,0],[6,0],[3,5],[9,5]], dtype=float)
points   = np.vstack([
    centers[i] + np.random.normal(0, 1.5, (n_per, 2))
    for i in range(k)
])
inst = MSSCInstance(points=points, k=k, name="overlapping_k4")

sol = jmeans(inst, n_restarts=5, seed=42)
print(f"j-means cost: {sol.cost:.4f}")

# Build Cluster objects from labels
rmp = RestrictedMasterProblem(inst)
clusters = []
for label in range(k):
    indices = tuple(np.where(sol.labels == label)[0].tolist())
    if len(indices) == 0:
        continue
    pts = inst.points[list(indices)]
    centroid = pts.mean(axis=0)
    cost = float(np.sum(np.sum((pts - centroid)**2, axis=1)))
    c = Cluster(indices=indices, centroid=centroid, cost=cost)
    clusters.append(c)
    print(f"  Cluster {label}: {len(indices)} points, cost={cost:.4f}")

print(f"\nTotal initial cost: {sum(c.cost for c in clusters):.4f}")

rmp.add_columns(clusters)

lp_obj, z_vals, sigma = rmp.solve()
print(f"\nAfter initial columns:")
print(f"  LP obj:      {lp_obj:.4f}")
print(f"  Num columns: {len(z_vals)}")
print(f"  z values:    {np.round(z_vals, 4)}")
print(f"  Fractional:  {[round(z,4) for z in z_vals if 1e-4 < z < 1-1e-4]}")
print(f"  sigma:       {sigma:.4f}")

# Check what's available after solve for lambdas
print(f"\nRMP attrs after solve: {[a for a in dir(rmp) if not a.startswith('_')]}")
