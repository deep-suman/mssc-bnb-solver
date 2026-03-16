import gurobipy as gp
from gurobipy import GRB

# Simple LP where we KNOW the exact dual values:
# min  x
# s.t. x >= 3   → dual should be +1 (tightening this increases obj)
#      x <= 10  → dual should be  0 (not binding)

m = gp.Model()
m.setParam("OutputFlag", 0)
x  = m.addVar(lb=0)
c1 = m.addConstr(x >= 3,  name="geq")
c2 = m.addConstr(x <= 10, name="leq")
m.setObjective(x, GRB.MINIMIZE)
m.optimize()

print(f"obj       = {m.ObjVal}")       # expect 3
print(f"x         = {x.X}")            # expect 3
print(f"Pi(geq)   = {c1.Pi}")          # what sign does Gurobi give?
print(f"Pi(leq)   = {c2.Pi}")          # expect 0
