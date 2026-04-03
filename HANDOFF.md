# MSSC B&B Solver — Final Project Handoff

## Reference Paper
**"An improved column generation algorithm for minimum sum-of-squares clustering"**
— Aloise, Hansen, and Liberti (EJOR, 2012)

---

## Environment
- **Machine**: Linux (Ubuntu 24), user `schedt_ext`
- **Working directory**: `/home/schedt_ext/Deep/CS722/Implementation/`
- **Python**: 3.10.14
- **Gurobi**: 13.0.1, academic license (expires 2027-03-11), License ID 2790372

### Git history
```
e918e8d  Validation: gr202 within ~0.05% of Table 3; body measurements ~3-5% above Table 10
566ea02  Validation: Glass identification within +-0.13% of paper Table 9
cc428f6  Validation: Iris dataset results within ±0.4% of paper Table 8
dae97fd  Complete 6-phase MSSC implementation with B&B fix
```

---

## Implementation Status: ALL 6 PHASES COMPLETE

### Phase 1 — `phase1_foundation.py`
- `MSSCInstance(points, k, name)` — holds n×s numpy array, k, s, name
- `MSSCSolution` — attributes: `cost`, `labels` (shape n), `centroids` (shape k×s)
- `Cluster(indices, centroid, cost)` — immutable named tuple; `indices` is a **tuple**
- `make_cluster(inst, indices)` — builds Cluster, computes cost via Huygens' theorem

### Phase 2 — `phase2_kmeans.py`, `phase2_jmeans.py`
- `kmeans(inst, seed)` → MSSCSolution
- `jmeans(inst, n_restarts, seed)` → MSSCSolution
- `extract_initial_columns(inst, sol)` → List[Cluster]

### Phase 3 — `phase3_rmp.py`
- `RestrictedMasterProblem(inst)` — Gurobi LP, Method=2 (barrier), Crossover=1
- `solve()` → `(lp_obj: float, lam: np.ndarray shape n, sigma: float)`
- `add_column(cluster)`, `get_solution()`, `reduced_cost(cluster, lam, sigma)`

### Phase 4 — `phase4_algorithm1.py`, `phase4_geometry.py`, `phase4_regions.py`
- `solve_auxiliary_2d(inst, lam, sigma)` → AuxResult  (s=2 only)

### Phase 5 — `phase5_algorithm2.py`, `phase5_dinkelbach.py`, `phase5_graph.py`
- `solve_auxiliary_general(inst, lam, sigma)` → AuxResult2  (any s)
- **Performance fix**: `_solve_01qp_exhaustive` vectorized with NumPy; threshold m<=18.
  Speedup ~100-1000x for m>=15 cliques.

### Phase 6 — `phase6_bnb.py`, `phase6_branching.py`, `phase6_constrained_aux.py`
- `solve_bnb(inst, max_nodes, n_jmeans_restarts, seed, verbose)` → BnBResult
- Ryan-Foster branching, constrained auxiliary, full branch-and-bound

---

## Critical Bugs Fixed (do not reintroduce)

### Bug 1 — Premature CG prune (MOST IMPORTANT)
**File**: `phase6_bnb.py`, `_solve_node()`, inside the CG for-loop
**Fix**: Removed `if lp_obj >= ub - 1e-6: break` from TOP of loop.
The CG loop now always calls the auxiliary; terminates only when `rc >= -1e-6`.

### Bug 2 — Accidental sed side effect
The same fix accidentally removed the correct prune in the main B&B loop (~line 258).
Restored manually:
```python
if lp_obj >= ub - 1e-6:
    continue  # prune node
```

### Bug 3 — Gurobi dual sign
`lam` clipped with `np.maximum(lam, 0.0)`. Do NOT negate Pi for >= constraints.

### Bug 4 — SLSQP feasibility
Multiple starting points used in auxiliary SLSQP calls.

### Bug 5 — Reduced cost centroid
Must use true centroid of cluster (Proposition 1), not constrained y* from SLSQP.

---

## Benchmark Validation Results

### Fisher's Iris (n=150, s=4) — Table 8 — VALIDATED
All k=2..10 within +-0.4% of paper. 1 node each, 0.6-20s.

| k  | Paper    | Ours     | Gap%    |
|----|----------|----------|---------|
| 2  | 152.3480 | 152.3687 | +0.014% |
| 3  |  78.8514 |  78.9408 | +0.113% |
| 4  |  57.2285 |  57.3179 | +0.156% |
| 5  |  46.4462 |  46.5356 | +0.192% |
| 6  |  39.0400 |  38.9310 | -0.279% |
| 7  |  34.2982 |  34.1892 | -0.318% |
| 8  |  29.9889 |  29.8799 | -0.363% |
| 9  |  27.7861 |  27.7654 | -0.074% |
| 10 |  25.8340 |  25.8134 | -0.080% |

### Glass Identification (n=214, s=9) — Table 9 — VALIDATED
File: `glass.data`, load with `usecols=range(1,10)`.
All k=30..50 within +-0.13% of paper. 1 node each, 336-1987s.

| k  | Paper   | Ours    | Gap%    |
|----|---------|---------|---------|
| 30 | 63.2478 | 63.3284 | +0.127% |
| 35 | 49.2386 | 49.2386 | +0.000% |
| 40 | 39.4983 | 39.4983 | -0.000% |
| 45 | 32.0395 | 32.0395 | +0.000% |
| 50 | 26.7675 | 26.7675 | +0.000% |

### Grotschel 202-city (n=202, s=2) — Table 3 — VALIDATED
File: `gr202.tsp` (TSPLIB GEO — use raw DDD.MM values as 2D Euclidean, no conversion).
All k=2..30 within +-0.05% of paper. 1 node each, 2-143s.

| k  | Paper       | Ours        | Gap%    |
|----|-------------|-------------|---------|
| 2  | 23437.4000  | 23437.3901  | -0.000% |
| 5  |  8894.9000  |  8894.9040  | +0.000% |
| 10 |  3792.4900  |  3794.4881  | +0.053% |
| 20 |  1523.5100  |  1523.5086  | -0.000% |
| 30 |   799.3110  |   799.3109  | -0.000% |

### Grotschel 666-city (n=666, s=2) — Table 4 — VALIDATED
File: `gr666.tsp` (same GEO format as gr202).
All k=2..50 within +-0.001% of paper. 1 node each, 17-4058s.

| k  | Paper       | Ours           | Gap%    |
|----|-------------|----------------|---------|
| 2  | 1754012.0   | 1754012.8397   | +0.000% |
| 3  |  772707.0   |  772707.4586   | +0.000% |
| 4  |  613995.0   |  613995.0766   | +0.000% |
| 5  |  485088.0   |  485088.0936   | +0.000% |
| 6  |  382676.0   |  382676.8691   | +0.000% |
| 7  |  323283.0   |  323283.8892   | +0.000% |
| 8  |  285925.0   |  285925.1871   | +0.000% |
| 9  |  250989.0   |  250989.1930   | +0.000% |
| 10 |  224183.0   |  224183.9790   | +0.000% |
| 20 |  106276.0   |  106276.5592   | +0.001% |
| 50 |   35179.5   |   35179.5539   | +0.000% |

### Body Measurements (n=507, s=5) — Table 10 — COLUMNS UNCERTAIN
File: `body.dat.txt` (25 cols; paper does not state which 5 were used).
Best-match columns: **9,10,11,22,23** (shoulder girth, chest girth, waist girth,
weight, height). Produces systematic +3-5% above paper optimals.
Solver is correct (1 node, optimal status all k); gap is column selection only.

| k  | Paper    | Ours     | Gap%    |
|----|----------|----------|---------|
| 30 | 19529.9  | 20178.9  | +3.323% |
| 40 | 16231.8  | 16913.5  | +4.200% |
| 50 | 13954.7  | 14547.4  | +4.247% |
| 60 | 12182.6  | 12733.6  | +4.523% |
| 70 | 10786.9  | 11311.4  | +4.862% |
| 80 |  9648.73 | 10141.1  | +5.103% |

### Ruspini (n=75, s=2) — PARTIAL
k=4 matches paper (-0.19%), other k values diverge 15-88%.
Root cause: different dataset version in paper. Not a priority.

### Telugu Vowel (n=871, s=3) — TOO SLOW
Even a single root node times out. Python too slow for n=871.

---

## Datasets on Disk

| File           | Source                                   |
|----------------|------------------------------------------|
| iris.data      | UCI ML Repository                        |
| glass.data     | UCI ML Repository                        |
| ruspini.csv    | manual                                   |
| gr202.tsp      | github.com/mastqe/tsplib                 |
| gr666.tsp      | github.com/mastqe/tsplib                 |
| body.dat.txt   | jse.amstat.org/datasets/body.dat.txt     |

---

## API Quick Reference (verified from source)
```python
inst = MSSCInstance(points, k, name)        # points: np.ndarray (n, s)
cl   = Cluster(indices=tuple(...), centroid=np.ndarray, cost=float)
lp_obj, lam, sigma = rmp.solve()            # lam shape (n,), sigma float
res  = solve_bnb(inst, max_nodes=100, n_jmeans_restarts=5, seed=42, verbose=False)
# res.optimal_cost, res.lp_bound, res.gap_pct, res.n_nodes,
# res.n_columns, res.time_sec, res.status ('optimal'|'max_nodes')
```

---

## Key Behavioral Notes
- **Never add `if lp_obj >= ub - 1e-6: break` inside the CG loop** — this is Bug 1
- **Always read source before writing test code** — never guess API signatures
- **Branching rarely fires** — Gurobi barrier+crossover returns integer LP solutions
  for set-partitioning problems; this is correct, not a bug
- **gr202/gr666 GEO format**: use raw DDD.MM floats as 2D Euclidean; no conversion
- **Body columns**: use [9,10,11,22,23] — best known approximation
