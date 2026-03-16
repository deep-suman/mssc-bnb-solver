import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

# Load Ruspini data
data = np.loadtxt('/home/schedt_ext/Deep/CS722/Implementation/ruspini.csv', delimiter=',')
print(f"Data shape: {data.shape}")
print(f"Data range: x=[{data[:,0].min():.1f}, {data[:,0].max():.1f}]  y=[{data[:,1].min():.1f}, {data[:,1].max():.1f}]")

# Paper Table 2 optimal values
paper_opts = {
    2: 0.893378e+05,
    3: 0.510634e+05,
    4: 0.128810e+05,
    5: 0.101267e+05,
    6: 0.857541e+04,
    7: 0.712620e+04,
    8: 0.614964e+04,
    9: 0.518165e+04,
   10: 0.444628e+04,
}

print(f"\n{'k':>3}  {'paper_opt':>14}  {'our_cost':>12}  {'gap%':>8}")
print("-" * 45)
for k, fopt in paper_opts.items():
    inst = MSSCInstance(points=data, k=k, name=f"ruspini_k{k}")
    res  = solve_bnb(inst, verbose=False, n_jmeans_restarts=10, seed=42)
    gap  = 100.0 * (res.optimal_cost - fopt) / fopt
    print(f"{k:>3}  {fopt:>14.2f}  {res.optimal_cost:>12.4f}  {gap:>+8.3f}%")
