import numpy as np
from phase1_foundation import MSSCInstance
from phase2_jmeans import jmeans

centers = np.array([[0.,0.], [50.,0.], [25.,43.]])
spread  = np.array([
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
], dtype=float)
points = np.vstack([centers[0]+spread, centers[1]+spread, centers[2]+spread])

inst   = MSSCInstance(points=points, k=3, name="test")
jm_sol = jmeans(inst, n_restarts=10, seed=42)

print(f"j-means cost : {jm_sol.cost:.4f}  (true optimal=54.0)")
print(f"Labels       : {jm_sol.labels}")
print(f"Centroids    :\n{jm_sol.centroids}")
for j in range(3):
    members = np.where(jm_sol.labels == j)[0]
    print(f"Cluster {j}: entities {members.tolist()}")
