import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase6_branching import find_branching_pair, is_integer_solution
from phase6_bnb import solve_bnb

# 4 points forming a square — k=2
# Three equally-valid partitions:
#   A: {0,1} and {2,3}  (left-right split)
#   B: {0,2} and {1,3}  (top-bottom split)
#   C: {0,3} and {1,2}  (diagonal split)
# LP can mix these fractionally → fractional solution

points = np.array([
    [0.0, 0.0],   # 0: bottom-left
    [1.0, 0.0],   # 1: bottom-right
    [0.0, 1.0],   # 2: top-left
    [1.0, 1.0],   # 3: top-right
])
inst = MSSCInstance(points=points, k=2, name="square_k2")

# Seed RMP with ALL three partitions as overlapping columns
rmp = RestrictedMasterProblem(inst)
rmp.add_column(make_cluster(inst, [0, 1]))   # partition A col 1
rmp.add_column(make_cluster(inst, [2, 3]))   # partition A col 2
rmp.add_column(make_cluster(inst, [0, 2]))   # partition B col 1
rmp.add_column(make_cluster(inst, [1, 3]))   # partition B col 2
rmp.add_column(make_cluster(inst, [0, 3]))   # partition C col 1
rmp.add_column(make_cluster(inst, [1, 2]))   # partition C col 2

obj, lam, sigma = rmp.solve()
z = rmp.get_solution()

print(f"LP obj    : {obj:.6f}")
print(f"z values  : {np.round(z, 4)}")
print(f"Fractional: {not is_integer_solution(z)}")

pair = find_branching_pair(rmp.columns, z)
print(f"Branch pair: {pair}")

# Now run full B&B
print("\n--- Full B&B ---")
res = solve_bnb(inst, verbose=True, max_nodes=50)
print(f"\nResult: {res}")
