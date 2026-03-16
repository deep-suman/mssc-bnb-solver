from phase1_foundation import make_synthetic_instance, make_cluster
from phase2_jmeans import jmeans, extract_initial_columns
from phase3_rmp import RestrictedMasterProblem

inst   = make_synthetic_instance(n=30, k=3, s=2, spread=0.5, seed=0)
jm_sol = jmeans(inst, n_restarts=3, seed=0)
cols   = extract_initial_columns(inst, jm_sol)

rmp = RestrictedMasterProblem(inst)
rmp.add_columns(cols)

extra = make_cluster(inst, [0, 1, 2, 3, 4, 5])
rmp.add_column(extra)

obj, lam, sigma = rmp.solve()
print(f"obj={obj:.4f}")
print(f"λ non-zero count: {(lam > 1e-6).sum()} / {inst.n}")
print(f"σ={sigma:.4f}")
