import numpy as np

pts = np.loadtxt("ruspini.csv", delimiter=",")

# The original Ruspini data uses larger coordinate values
# Paper fopt k=2 = 89337 >> our 45207
# Ratio suggests coordinates might be scaled by ~sqrt(2) or data is different
# Let's check: if we multiply coordinates by some factor f,
# costs scale by f^2. To get 89337 from 45207: f^2 = 89337/45207 = 1.976 → f≈1.41
ratio = 89337.8 / 45207.3
print(f"Cost ratio paper/ours: {ratio:.4f}")
print(f"Scale factor needed  : {ratio**0.5:.4f}")

# The actual Ruspini dataset has coordinates in range ~[20, 120]
# Let's check a known source: x scaled by 10?
pts_scaled = pts * ratio**0.5
total_var = ((pts_scaled - pts_scaled.mean(axis=0))**2).sum()
print(f"Scaled total variance: {total_var:.1f}")
print(f"Expected (> 89337)   : 103827 * {ratio:.4f} = {103827*ratio:.1f}")
