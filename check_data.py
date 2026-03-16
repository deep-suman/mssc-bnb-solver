import numpy as np

pts = np.loadtxt("ruspini.csv", delimiter=",")
print(f"Shape : {pts.shape}")
print(f"X range: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
print(f"Y range: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")

# The paper reports fopt=89337 for k=2
# With our data, max possible cost is bounded by total variance:
total_var = ((pts - pts.mean(axis=0))**2).sum()
print(f"Total variance (k=1 cost): {total_var:.1f}")
print(f"If total_var << 89337, our data is wrong")
