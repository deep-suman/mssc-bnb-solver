import numpy as np
from phase1_foundation import MSSCInstance, Cluster, make_cluster
from phase3_rmp import RestrictedMasterProblem
from phase2_jmeans import jmeans, extract_initial_columns
from phase4_algorithm1 import solve_auxiliary_2d

# Same overlapping instance
np.random.seed(123)
k, n_per = 4, 15
centers  = np.array([[0,0],[6,0],[3,5],[9,5]], dtype=float)
points   = np.vstack([
    centers[i] + np.random.normal(0, 1.5, (n_per, 2))
    for i in range(k)
])
inst = MSSCInstance(points=points, k=k, name="overlapping_k4")

# Build RMP with initial columns
sol      = jmeans(inst, n_restarts=5, seed=42)
init_cols = extract_initial_columns(inst, sol)
rmp      = RestrictedMasterProblem(inst)
rmp.add_columns(init_cols)

print(f"Initial columns added: {len(rmp.columns)}")

# Solve LP and get duals
lp_obj, lam, sigma = rmp.solve()
z_vals = rmp.get_solution()

print(f"LP obj:    {lp_obj:.4f}")
print(f"sigma:     {sigma:.4f}")
print(f"lam range: [{lam.min():.4f}, {lam.max():.4f}]")
print(f"z values:  {np.round(z_vals, 4)}")
print(f"z integer: {all(abs(z - round(z)) < 1e-4 for z in z_vals)}")

# Check reduced costs of existing columns
print(f"\nReduced costs of initial columns:")
for i, cl in enumerate(rmp.columns):
    rc = rmp.reduced_cost(cl, lam, sigma)
    print(f"  col {i}: indices={cl.indices[:5]}... rc={rc:.6f}")

# Now run the auxiliary problem and check what it returns
print(f"\nAuxiliary problem result:")
aux = solve_auxiliary_2d(inst, lam, sigma)
print(f"  reduced_cost: {aux.reduced_cost:.6f}")
print(f"  cluster indices: {aux.cluster.indices if aux.cluster else None}")
print(f"  Is negative rc: {aux.reduced_cost < -1e-6}")

# Manually verify: check reduced cost of a few random subsets
print(f"\nBrute-force check (random subsets of size 2-5):")
np.random.seed(0)
best_rc = 0.0
best_subset = None
for _ in range(5000):
    size = np.random.randint(2, 6)
    idx  = tuple(sorted(np.random.choice(inst.n, size, replace=False).tolist()))
    pts  = inst.points[list(idx)]
    centroid = pts.mean(axis=0)
    cost = float(np.sum((pts - centroid)**2))
    rc   = cost + sigma - sum(lam[i] for i in idx)
    if rc < best_rc:
        best_rc     = rc
        best_subset = idx
print(f"  Best rc found: {best_rc:.6f}")
print(f"  Best subset:   {best_subset}")
