import numpy as np
from phase1_foundation import MSSCInstance
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem

np.random.seed(0)
n      = 20
k      = 4
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
points = np.column_stack([np.cos(angles), np.sin(angles)])
points += np.random.normal(0, 0.01, points.shape)

inst   = MSSCInstance(points=points, k=k, name="circle")
jm_sol = jmeans(inst, n_restarts=5, seed=0)
cols   = extract_initial_columns(inst, jm_sol)

print(f"j-means cost : {jm_sol.cost:.6f}")
print(f"Num columns  : {len(cols)}")

rmp = RestrictedMasterProblem(inst)
rmp.add_columns(cols)

obj, lam, sigma = rmp.solve()
z = rmp.get_solution()

print(f"LP obj       : {obj:.6f}")
print(f"z values     : {np.round(z, 4)}")
print(f"All integer? : {all(abs(zi - round(zi)) < 1e-4 for zi in z)}")
print(f"sigma        : {sigma:.6f}")
print(f"lambda range : [{lam.min():.4f}, {lam.max():.4f}]")
