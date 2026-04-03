"""
Find which 5 columns of body.dat.txt match the paper's Table 10 costs.
Strategy: run j-means heuristic (upper bound) on candidate column sets at k=30.
Paper target: fopt=19529.9. Our UB should be within ~5% if columns are right.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_foundation import MSSCInstance
from phase2_jmeans import jmeans

data = np.loadtxt('body.dat.txt')
TARGET = 19529.9

# Column metadata (from body.txt description)
COL_NAMES = [
    'biacromial_d','biiliac_d','bitrochanteric_d','chest_depth','chest_d',   # 0-4  skeletal
    'elbow_d','wrist_d','knee_d','ankle_d',                                   # 5-8  skeletal
    'shoulder_g','chest_g','waist_g','navel_g','hip_g',                       # 9-13 girth
    'thigh_g','bicep_g','forearm_g','knee_g','calf_g','ankle_g','wrist_g',   # 14-20 girth
    'age','weight','height','gender'                                           # 21-24
]

# Candidate sets to try: most plausible 5-column subsets
candidates = {
    'cols0-4  (skeletal 1-5)':      [0,1,2,3,4],
    'cols5-8+0 (skeletal mix)':     [0,5,6,7,8],
    'cols9-13 (girths 1-5)':        [9,10,11,12,13],
    'cols10-14 (girths 2-6)':       [10,11,12,13,14],
    'cols11-15 (girths 3-7)':       [11,12,13,14,15],
    'cols9,10,12,13,14':            [9,10,12,13,14],
    'cols9,11,12,13,14':            [9,11,12,13,14],
    'wgt+hgt+3girths[9,10,11]':     [9,10,11,22,23],
    'cols0-3+22 (skel+weight)':     [0,1,2,3,22],
    'cols0-4 scaled x10':           'scale',   # special case
}

print(f"{'Columns':<35}  {'var_sum':>10}  {'jmeans_UB':>12}  {'ratio_to_target':>16}")
print("-"*80)
for name, cols in candidates.items():
    if cols == 'scale':
        pts = data[:, [0,1,2,3,4]] * 10
        var_sum = float(np.var(pts, axis=0).sum())
    else:
        pts = data[:, cols]
        var_sum = float(np.var(pts, axis=0).sum())
    
    inst = MSSCInstance(pts, 30, name='body_k30')
    sol  = jmeans(inst, n_restarts=3, seed=42)
    ratio = sol.cost / TARGET
    print(f"{name:<35}  {var_sum:>10.2f}  {sol.cost:>12.2f}  {ratio:>15.3f}x")
