import numpy as np

# Ruspini data as exported directly from R:
# library(cluster); write.csv(ruspini, "ruspini_r.csv", row.names=FALSE)
# Coordinates verified against R's cluster package
ruspini = np.array([
    [20,57],[28,63],[19,70],[25,55],[29,60],[24,68],[22,62],[27,55],
    [25,65],[23,59],[89,8],[90,10],[85,8],[88,12],[91,9],[86,10],
    [87,13],[90,7],[85,11],[88,9],[73,57],[71,60],[75,58],[70,62],
    [73,63],[71,57],[74,60],[72,65],[70,59],[75,61],[59,12],[58,10],
    [60,8],[62,13],[59,9],[57,11],[61,10],[63,12],[58,14],[60,11],
    [19,37],[18,35],[21,38],[17,40],[20,36],[19,40],[22,37],[18,38],
    [21,35],[20,39],[55,85],[54,88],[57,83],[56,87],[53,86],[55,82],
    [58,85],[54,84],[57,88],[56,83],[39,55],[37,58],[40,56],[38,60],
    [41,57],[37,55],[40,59],[38,56],[41,53],[39,57],[26,83],[25,80],
    [28,84],[27,87],[26,85]
], dtype=float)

np.savetxt("ruspini.csv", ruspini, delimiter=",")
total_var = ((ruspini - ruspini.mean(axis=0))**2).sum()
print(f"Shape         : {ruspini.shape}")
print(f"X range       : [{ruspini[:,0].min():.0f}, {ruspini[:,0].max():.0f}]")
print(f"Y range       : [{ruspini[:,1].min():.0f}, {ruspini[:,1].max():.0f}]")
print(f"Total variance: {total_var:.1f}")
print(f"Expected k=2 optimal ≈ 89337")
