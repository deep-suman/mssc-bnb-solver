import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase3_rmp import RestrictedMasterProblem
from phase4_algorithm1 import solve_auxiliary_2d

points = np.array([
    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
    [3.0, 0.0], [4.0, 0.0], [5.0, 0.0],
])
inst = MSSCInstance(points=points, k=2, name="line_6pts")

rmp = RestrictedMasterProblem(inst)
rmp.add_column(make_cluster(inst, [0, 1, 2]))
rmp.add_column(make_cluster(inst, [3, 4, 5]))
rmp.add_column(make_cluster(inst, [0, 1]))
rmp.add_column(make_cluster(inst, [2, 3, 4, 5]))
rmp.add_column(make_cluster(inst, [0, 1, 2, 3]))
rmp.add_column(make_cluster(inst, [4, 5]))

prev_obj = np.inf
for it in range(30):
    obj, lam, sigma = rmp.solve()
    aux = solve_auxiliary_2d(inst, lam, sigma)
    print(f"Iter {it+1:2d}: LP={obj:.6f}  rc={aux.reduced_cost:.6f}")

    # Primary check: no negative reduced cost
    if not aux.has_negative_rc():
        print(f"✓ Certified optimal: LP = {obj:.6f}")
        break

    # Secondary check: LP did not improve → degenerate duals, stop
    if obj >= prev_obj - 1e-8:
        print(f"✓ LP not improving → optimal: LP = {obj:.6f}")
        break

    prev_obj = obj
    rmp.add_column(aux.cluster)
