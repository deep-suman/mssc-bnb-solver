import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase2_jmeans import jmeans
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d
from phase6_branching import find_branching_pair, is_integer_solution

np.random.seed(7)
n = 12
A = np.random.normal([0, 0], 0.2, (n//2, 2))
B = np.random.normal([3, 0], 0.2, (n//2, 2))
points = np.vstack([A, B])
np.random.shuffle(points)

inst     = MSSCInstance(points=points, k=2, name="two_groups")
jm_sol   = jmeans(inst, n_restarts=10, seed=0)
bad_col1 = make_cluster(inst, list(range(0, n, 2)))
bad_col2 = make_cluster(inst, list(range(1, n, 2)))
print(f"Bad UB    : {bad_col1.cost + bad_col2.cost:.4f}")
print(f"j-means UB: {jm_sol.cost:.4f}")

rmp = RestrictedMasterProblem(inst)
rmp.add_column(bad_col1)
rmp.add_column(bad_col2)

prev_obj = np.inf
lp_moved = False
print(f"\nColumn generation:")
for it in range(50):
    obj, lam, sigma = rmp.solve()
    aux = solve_auxiliary_2d(inst, lam, sigma)
    print(f"  Iter {it+1:2d}: LP={obj:.4f}  rc={aux.reduced_cost:.6f}  "
          f"cols={len(rmp.columns)}")

    if not aux.has_negative_rc():
        print(f"  → No negative rc — LP optimal")
        break
    if lp_moved and obj >= prev_obj - 1e-8:
        print(f"  → LP not improving")
        break
    if obj < prev_obj - 1e-8:
        lp_moved = True
    if aux.cluster and aux.cluster.indices not in rmp._col_index_sets:
        rmp.add_column(aux.cluster)
    prev_obj = obj

z = rmp.get_solution()
print(f"\nFinal LP  : {obj:.4f}")
print(f"z values  : {np.round(z, 4)}")
print(f"Fractional: {not is_integer_solution(z)}")
pair = find_branching_pair(rmp.columns, z)
print(f"Branch pair: {pair}")
