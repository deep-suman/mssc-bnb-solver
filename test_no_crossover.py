import numpy as np
import gurobipy as gp
from phase1_foundation import MSSCInstance, make_cluster
from phase3_rmp import RestrictedMasterProblem

points = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.],[5.,0.]])
inst   = MSSCInstance(points=points, k=2, name="test")

rmp = RestrictedMasterProblem(inst)
rmp.add_column(make_cluster(inst, [0,1,2]))
rmp.add_column(make_cluster(inst, [1,2,3]))
rmp.add_column(make_cluster(inst, [2,3,4]))
rmp.add_column(make_cluster(inst, [3,4,5]))
rmp.add_column(make_cluster(inst, [0,1,2,3]))
rmp.add_column(make_cluster(inst, [2,3,4,5]))

# Try with crossover disabled
rmp.model.setParam("Crossover", 0)
obj, lam, sigma = rmp.solve()
z = rmp.get_solution()

print(f"LP obj    : {obj:.6f}")
print(f"z values  : {np.round(z,4)}")
from phase6_branching import is_integer_solution, find_branching_pair
print(f"Fractional: {not is_integer_solution(z)}")
print(f"Branch pair: {find_branching_pair(rmp.columns, z)}")
