import numpy as np

centers = np.array([[0.,0.], [50.,0.], [25.,43.]])
spread  = np.array([
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
    [-2,-1],[2,-1],[0,2],[-1,1],[1,-1],
], dtype=float)

points = np.vstack([centers[0]+spread, centers[1]+spread, centers[2]+spread])
print(f"Total points: {len(points)}")
print(f"Points per cluster: 15 (not 5!)")

# Recompute true optimal with 15 points per cluster
true_opt = 0.0
for c in range(3):
    pts      = points[c*15:(c+1)*15]
    centroid = pts.mean(axis=0)
    cost     = ((pts - centroid)**2).sum()
    print(f"Cluster {c}: centroid={centroid}, cost={cost:.4f}")
    true_opt += cost

print(f"True optimal cost: {true_opt:.4f}")
# j-means found 162.0 → matches!
