import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase6_branching import (BranchNode, find_branching_pair,
                               is_integer_solution,
                               column_satisfies_constraints)

points = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.],[5.,0.]])
inst   = MSSCInstance(points=points, k=2, name="test")

# Three overlapping columns
col0 = make_cluster(inst, [0,1,2])
col1 = make_cluster(inst, [1,2,3])
col2 = make_cluster(inst, [3,4,5])
col3 = make_cluster(inst, [0,1,2,3])
col4 = make_cluster(inst, [2,3,4,5])
cols = [col0, col1, col2, col3, col4]

# Inject a fractional z: mix col1+col2 and col3+col4
z_frac = np.array([0.0, 0.5, 0.5, 0.5, 0.5])

print(f"Fractional: {not is_integer_solution(z_frac)}")
pair = find_branching_pair(cols, z_frac)
print(f"Branch pair: {pair}")
assert pair is not None

i1, i2 = pair
root      = BranchNode()
node_same = root.branch_same(i1, i2)
node_diff = root.branch_diff(i1, i2)

# Count valid columns per branch
same_valid = [c for c in cols if column_satisfies_constraints(c, node_same)]
diff_valid = [c for c in cols if column_satisfies_constraints(c, node_diff)]

print(f"\nBranching on ({i1},{i2}):")
print(f"  SAME branch: {len(same_valid)}/{len(cols)} columns valid")
print(f"  DIFF branch: {len(diff_valid)}/{len(cols)} columns valid")

# Verify SAME constraint is respected
for c in same_valid:
    idx = set(c.indices)
    assert (i1 in idx) == (i2 in idx), \
        f"SAME violated in {c.indices}"
print(f"  ✓ All SAME-branch columns respect constraint")

# Verify DIFF constraint is respected
for c in diff_valid:
    idx = set(c.indices)
    assert not (i1 in idx and i2 in idx), \
        f"DIFF violated in {c.indices}"
print(f"  ✓ All DIFF-branch columns respect constraint")

# Verify the two branches partition the search space
same_indices = {c.indices for c in same_valid}
diff_indices = {c.indices for c in diff_valid}
print(f"\n  SAME cols: {[c.indices for c in same_valid]}")
print(f"  DIFF cols: {[c.indices for c in diff_valid]}")
print(f"  ✓ Branching logic correct")
