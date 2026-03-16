import numpy as np
from phase1_foundation import MSSCInstance, make_cluster
from phase3_rmp import RestrictedMasterProblem
from phase6_branching import find_branching_pair, is_integer_solution
from phase6_bnb import solve_bnb, _solve_node, _extract_labels
from phase6_branching import BranchNode

# 6 points, k=2
# We manually inject columns that FORCE a fractional LP solution:
# No single pair of columns covers all 6 entities exactly once,
# so the LP must fractionally combine multiple columns.
points = np.array([
    [0.,0.],[1.,0.],[2.,0.],
    [0.,1.],[1.,1.],[2.,1.]
])
inst = MSSCInstance(points=points, k=2, name="force_frac")

# Three partial columns — each covers 4 of 6 entities
# No two of them together cover all 6 exactly → LP must mix
rmp = RestrictedMasterProblem(inst)
rmp.add_column(make_cluster(inst, [0,1,3,4]))  # top-left square
rmp.add_column(make_cluster(inst, [1,2,4,5]))  # top-right square
rmp.add_column(make_cluster(inst, [0,2,3,5]))  # diagonal
rmp.add_column(make_cluster(inst, [0,1,2]))    # bottom row
rmp.add_column(make_cluster(inst, [3,4,5]))    # top row

obj, lam, sigma = rmp.solve()
z    = rmp.get_solution()
frac = not is_integer_solution(z)
pair = find_branching_pair(rmp.columns, z)

print(f"LP obj     : {obj:.6f}")
print(f"z values   : {np.round(z,4)}")
print(f"Fractional : {frac}")
print(f"Branch pair: {pair}")

if frac and pair is not None:
    i1, i2 = pair
    print(f"\nBranching on ({i1},{i2}):")

    # SAME branch
    node_same = BranchNode(same_pairs=[(i1,i2)])
    valid_same = [cl for cl in rmp.columns
                  if __import__('phase6_branching')
                  .column_satisfies_constraints(cl, node_same)]
    print(f"  SAME branch: {len(valid_same)} valid columns")

    # DIFF branch
    node_diff = BranchNode(diff_pairs=[(i1,i2)])
    valid_diff = [cl for cl in rmp.columns
                  if __import__('phase6_branching')
                  .column_satisfies_constraints(cl, node_diff)]
    print(f"  DIFF branch: {len(valid_diff)} valid columns")

    print(f"\n--- Full B&B ---")
    res = solve_bnb(inst, verbose=True, max_nodes=20,
                    n_jmeans_restarts=5, seed=0)
    print(f"Result: {res}")
else:
    print("LP is integer — no branching needed")
