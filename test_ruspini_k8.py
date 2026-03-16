import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

pts  = np.loadtxt("ruspini.csv", delimiter=",")
inst = MSSCInstance(points=pts, k=8, name="Ruspini_k8")

# Paper reports: fopt=6149.64, gap exists, 3 B&B nodes needed
res = solve_bnb(inst, verbose=True, max_nodes=50,
                n_jmeans_restarts=10, seed=42)
print(f"\nResult : {res}")
print(f"Paper  : fopt=6149.64")
print(f"Our UB : {res.optimal_cost:.2f}")
