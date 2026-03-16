import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase6_branching import find_branching_pair, is_integer_solution
from phase6_bnb import solve_bnb

# 8 points: two squares offset by (2,0)
# k=2: two valid partitions of equal cost
# LP must mix them → fractional solution
points = np.array([
    # Square 1 centered at (0,0)
    [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
    # Square 2 centered at (4,0)  
    [3.0, -1.0],  [5.0, -1.0], [5.0, 1.0], [3.0,  1.0],
])
inst = MSSCInstance(points=points, k=2, name="two_squares")

# Manually seed RMP with ALL valid 2-partitions
rmp = RestrictedMasterProblem(inst)

# Natural partition
rmp.add_column(make_cluster(inst, [0,1,2,3]))
rmp.add_column(make_cluster(inst, [4,5,6,7]))

# Cross partitions (equal cost due to symmetry)
rmp.add_column(make_cluster(inst, [0,1,2,4]))
rmp.add_column(make_cluster(inst, [3,5,6,7]))
rmp.add_column(make_cluster(inst, [0,1,3,4]))
rmp.add_column(make_cluster(inst, [2,5,6,7]))

obj, lam, sigma = rmp.solve()
z    = rmp.get_solution()
frac = not is_integer_solution(z)
pair = find_branching_pair(rmp.columns, z)

print(f"LP obj     : {obj:.4f}")
print(f"z values   : {np.round(z,4)}")
print(f"Fractional : {frac}")
print(f"Branch pair: {pair}")

print(f"\n--- Full B&B ---")
res = solve_bnb(inst, verbose=True, max_nodes=50,
                n_jmeans_restarts=10, seed=0)
print(f"\nResult: {res}")
