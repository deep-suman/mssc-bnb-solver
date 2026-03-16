import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

# Load Iris — 4 numeric columns, skip last label column and empty lines
data = []
with open('iris.data') as f:
    for line in f:
        line = line.strip()
        if line:
            vals = line.split(',')
            data.append([float(v) for v in vals[:4]])
data = np.array(data)
print(f"Data shape: {data.shape}")
print(f"Data range: min={data.min():.2f}  max={data.max():.2f}")

# Paper Table 8 optimal values (Fisher's Iris, n=150, s=4)
paper_opts = {
    2:  152.348,
    3:   78.8514,
    4:   57.2285,
    5:   46.4462,
    6:   39.0400,
    7:   34.2982,
    8:   29.9889,
    9:   27.7861,
   10:   25.834,
}

print(f"\n{'k':>3}  {'paper_opt':>12}  {'our_cost':>12}  {'gap%':>8}  {'nodes':>6}  {'time':>6}")
print("-" * 58)
for k, fopt in paper_opts.items():
    inst = MSSCInstance(points=data, k=k, name=f"iris_k{k}")
    res  = solve_bnb(inst, verbose=False, n_jmeans_restarts=10, seed=42)
    gap  = 100.0 * (res.optimal_cost - fopt) / fopt
    print(f"{k:>3}  {fopt:>12.4f}  {res.optimal_cost:>12.4f}  {gap:>+8.3f}%  "
          f"{res.n_nodes:>6}  {res.time_sec:>5.1f}s")
