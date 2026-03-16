import numpy as np
import gurobipy as gp
from gurobipy import GRB
from phase1_foundation import MSSCInstance, make_cluster

# Build the LP manually with ALL possible 2-clusters for 6 points
# If the LP has a fractional optimal, we'll find it
points = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.],[5.,0.]])
inst   = MSSCInstance(points=points, k=2, name="test")
n      = inst.n

# Generate all non-trivial subsets of size 2,3,4 as clusters
from itertools import combinations
from phase1_foundation import make_cluster

all_cols = []
for size in range(1, n):
    for idx in combinations(range(n), size):
        all_cols.append(make_cluster(inst, list(idx)))

print(f"Total columns: {len(all_cols)}")

# Build LP with ALL columns
m = gp.Model()
m.setParam("OutputFlag", 0)
m.setParam("Method", 2)       # barrier
m.setParam("Crossover", 0)    # no crossover → interior point

cover  = [m.addConstr(gp.LinExpr() >= 1.0) for _ in range(n)]
card   = m.addConstr(gp.LinExpr() <= 2.0)
zvars  = []

for cl in all_cols:
    col = gp.Column()
    for i in cl.indices:
        col.addTerms(1.0, cover[i])
    col.addTerms(1.0, card)
    zvars.append(m.addVar(obj=cl.cost, lb=0.0, column=col))

m.ModelSense = GRB.MINIMIZE
m.update()
m.optimize()

z = np.array([v.X for v in zvars])
frac_mask = (z > 1e-4) & (z < 1-1e-4)
print(f"LP obj      : {m.ObjVal:.6f}")
print(f"Nonzero z   : {np.round(z[z>1e-4], 4)}")
print(f"Fractional  : {frac_mask.any()}")
if frac_mask.any():
    frac_cols = [all_cols[i].indices for i in np.where(frac_mask)[0]]
    print(f"Frac columns: {frac_cols}")
