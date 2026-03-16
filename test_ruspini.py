import numpy as np
from phase1_foundation import MSSCInstance
from mssc_solver import solve_mssc

pts = np.loadtxt("ruspini.csv", delimiter=",")

# Paper Table 2 optimal values for reference:
# k=2: 89337.8  k=3: 51063.4  k=4: 12881.0  k=5: 10126.7
fopt = {2: 89337.8, 3: 51063.4, 4: 12881.0, 5: 10126.7}

for k in [2, 3, 4, 5]:
    inst = MSSCInstance(points=pts, k=k, name=f"Ruspini_k{k}")
    res  = solve_mssc(inst, verbose=False)
    gap_to_opt = 100*(res.upper_bound - fopt[k]) / fopt[k]
    print(f"k={k}: LP={res.lp_bound:.1f}  UB={res.upper_bound:.1f}  "
          f"gap={res.gap_pct():.2f}%  "
          f"UB_vs_opt={gap_to_opt:.2f}%  "
          f"time={res.time_sec:.2f}s")
