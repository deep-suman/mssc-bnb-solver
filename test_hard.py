import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

# Hard instance: points on a circle — no natural cluster structure,
# many equally-good partitions, j-means will struggle
np.random.seed(0)
n      = 20
k      = 4
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
points = np.column_stack([np.cos(angles), np.sin(angles)])

# Add tiny noise so points are not perfectly symmetric
points += np.random.normal(0, 0.01, points.shape)

inst = MSSCInstance(points=points, k=k, name="circle_n20_k4")
res  = solve_bnb(inst, verbose=True, max_nodes=500)
print(f"\nResult: {res}")
