import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d

points = np.array([
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [3.0, 0.0], [3.5, 0.0], [4.0, 0.0],
    [1.5, 0.0], [2.0, 0.0], [2.5, 0.0],
])
inst   = MSSCInstance(points=points, k=3, name="overlap")
jm_sol = jmeans(inst, n_restarts=3, seed=0)
ub     = jm_sol.cost
cols   = extract_initial_columns(inst, jm_sol)

rmp = RestrictedMasterProblem(inst)
rmp.add_columns(cols)

print(f"Initial UB = {ub:.4f}")
for it in range(40):
    obj, lam, sigma = rmp.solve()
    if obj >= ub - 1e-6:
        print(f"✓ LP obj = UB = {obj:.4f} → optimal at iter {it+1}")
        break
    aux = solve_auxiliary_2d(inst, lam, sigma)
    print(f"Iter {it+1:2d}: LP={obj:.4f}  rc={aux.reduced_cost:.6f}")
    if not aux.has_negative_rc():
        print(f"✓ Certified optimal at iter {it+1}")
        break
    if aux.cluster.indices in rmp._col_index_sets:
        print("! Cycling")
        break
    rmp.add_column(aux.cluster)
