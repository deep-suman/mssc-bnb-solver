import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

# Load glass data — col 0 is ID, cols 1-9 are features, col 10 is class
data = np.loadtxt('glass.data', delimiter=',', usecols=range(1, 10))
print(f"Data shape: {data.shape}")
print(f"Data range: min={data.min():.2f}  max={data.max():.2f}")

# Paper Table 9 optimal values (Glass, n=214, s=9)
# Only k>=30 which paper solves in reasonable time
paper_opts = {
    30: 0.632478e+02,
    35: 0.492386e+02,
    40: 0.394983e+02,
    45: 0.320395e+02,
    50: 0.267675e+02,
}

print(f"\n{'k':>4}  {'paper_opt':>12}  {'our_cost':>12}  {'gap%':>8}  {'nodes':>6}  {'time':>6}")
print("-" * 58)
for k, fopt in paper_opts.items():
    inst = MSSCInstance(points=data, k=k, name=f"glass_k{k}")
    res  = solve_bnb(inst, verbose=False, n_jmeans_restarts=10, seed=42)
    gap  = 100.0 * (res.optimal_cost - fopt) / fopt
    print(f"{k:>4}  {fopt:>12.4f}  {res.optimal_cost:>12.4f}  {gap:>+8.3f}%  "
          f"{res.n_nodes:>6}  {res.time_sec:>5.1f}s", flush=True)
