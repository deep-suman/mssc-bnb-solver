import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

data = []
with open('vowel_telugu.dat') as f:
    for i, line in enumerate(f):
        if i < 8: continue
        line = line.strip()
        if not line: continue
        vals = line.split()
        if len(vals) == 4 and vals[0] in '123456':
            data.append([float(vals[1]), float(vals[2]), float(vals[3])])
data = np.array(data)
print(f"Data shape: {data.shape}")

paper_opts = {
    80:  0.324801e+07,
    90:  0.285069e+07,
   100:  0.251058e+07,
}

print(f"\n{'k':>4}  {'paper_opt':>14}  {'our_cost':>14}  {'gap%':>8}  {'nodes':>6}  {'time':>6}")
print("-" * 62)
for k, fopt in paper_opts.items():
    inst = MSSCInstance(points=data, k=k, name=f"vowel_k{k}")
    # Use max_nodes=1 — just root node, no branching, time-boxed
    res  = solve_bnb(inst, verbose=False, n_jmeans_restarts=5,
                     seed=42, max_nodes=1)
    gap  = 100.0 * (res.optimal_cost - fopt) / fopt
    print(f"{k:>4}  {fopt:>14.2f}  {res.optimal_cost:>14.4f}  {gap:>+8.3f}%  "
          f"{res.n_nodes:>6}  {res.time_sec:>5.1f}s", flush=True)
