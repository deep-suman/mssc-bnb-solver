from phase1_foundation import make_synthetic_instance, make_cluster
from phase3_rmp import RestrictedMasterProblem
import numpy as np

# Small instance, manually controlled
inst = make_synthetic_instance(n=9, k=3, s=2, spread=2.0, seed=5)

rmp = RestrictedMasterProblem(inst)

# Add multiple overlapping columns so LP cannot trivially sit at integer solution
rmp.add_column(make_cluster(inst, [0,1,2]))
rmp.add_column(make_cluster(inst, [3,4,5]))
rmp.add_column(make_cluster(inst, [6,7,8]))
rmp.add_column(make_cluster(inst, [0,1,3]))   # overlapping
rmp.add_column(make_cluster(inst, [2,4,5]))   # overlapping
rmp.add_column(make_cluster(inst, [6,7,8]))   # duplicate — ignored
rmp.add_column(make_cluster(inst, [0,3,6]))   # cross-cluster

obj, lam, sigma = rmp.solve()
print(f"obj   = {obj:.4f}")
print(f"σ     = {sigma:.4f}")
print(f"λ     = {np.round(lam, 4)}")
print(f"λ non-zero count: {(lam > 1e-6).sum()} / {inst.n}")
