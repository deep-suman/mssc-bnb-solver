import numpy as np
from phase1_foundation import MSSCInstance
from mssc_solver import solve_mssc

centers = np.array([[0.,0.], [50.,0.], [25.,43.]])
spread  = np.array([
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
], dtype=float)
points = np.vstack([centers[0]+spread, centers[1]+spread, centers[2]+spread])

# Correct true optimal: 15 points per cluster
true_opt = 0.0
for c in range(3):
    pts      = points[c*15:(c+1)*15]
    centroid = pts.mean(axis=0)
    true_opt += ((pts - centroid)**2).sum()
print(f"True optimal cost: {true_opt:.4f}")

inst = MSSCInstance(points=points, k=3, name="known_optimal")
res  = solve_mssc(inst, verbose=True)

gap_to_opt = 100*(res.upper_bound - true_opt) / true_opt
print(f"\nUB vs true optimal: {gap_to_opt:.4f}%")
assert abs(gap_to_opt) < 1e-4, f"Expected 0% gap, got {gap_to_opt:.4f}%"
assert res.gap_pct() < 1e-4,   f"LP gap should be 0%, got {res.gap_pct():.4f}%"
print("✓ Solver validated against known optimal")
