import subprocess
import numpy as np

# Use R directly to export the exact dataset
r_script = """
library(cluster)
write.table(ruspini, stdout(), sep=",", row.names=FALSE, col.names=FALSE)
"""
result = subprocess.run(["Rscript", "--vanilla", "-e", r_script],
                        capture_output=True, text=True)
if result.returncode != 0:
    print("R not available:", result.stderr)
else:
    lines = result.stdout.strip().split("\n")
    data  = np.array([list(map(float, l.split(","))) for l in lines])
    np.savetxt("ruspini.csv", data, delimiter=",")
    total_var = ((data - data.mean(axis=0))**2).sum()
    print(f"Shape         : {data.shape}")
    print(f"X range       : [{data[:,0].min():.0f}, {data[:,0].max():.0f}]")
    print(f"Y range       : [{data[:,1].min():.0f}, {data[:,1].max():.0f}]")
    print(f"Total variance: {total_var:.1f}")
    print("First 5 rows  :", data[:5])
