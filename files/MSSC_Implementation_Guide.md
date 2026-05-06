# MSSC B&B Solver — Implementation Guide

**CS722 Optimization Algorithms | Based on Aloise, Hansen, Liberti (EJOR 2012)**

---

## Environment Setup

```bash
# Python 3.10.14 + Gurobi 13.0.1 required
# Gurobi academic license must be active

# Verify environment
python --version          # 3.10.14
python -c "import gurobipy; print(gurobipy.gurobi.version())"  # (13,0,1)
python -c "import numpy; print(numpy.__version__)"
```

All implementation files live in one flat directory — no packages or installs beyond numpy and gurobipy.

---

## File Structure

```
Implementation/
│
│  ── Core Implementation (original, validated) ──
├── phase1_foundation.py          Data structures: MSSCInstance, Cluster, MSSCSolution
├── phase2_kmeans.py              k-means++ initialization (Lloyd's algorithm)
├── phase2_jmeans.py              j-means local search heuristic
├── phase3_rmp.py                 Gurobi LP master (RestrictedMasterProblem)
├── phase4_algorithm1.py          2D auxiliary: Algorithm 1 (disc intersection)
├── phase4_geometry.py            Disc intersection geometry helpers
├── phase4_regions.py             Region enumeration for 2D auxiliary
├── phase5_graph.py               Hypersphere intersection graph (original)
├── phase5_algorithm2.py          General auxiliary: Algorithm 2 (original)
├── phase5_dinkelbach.py          Dinkelbach's algorithm + vectorized 0-1 QP
├── phase6_branching.py           Ryan-Foster branching rule
├── phase6_constrained_aux.py     Constrained auxiliary for B&B nodes
├── phase6_bnb.py                 Main B&B solver (solve_bnb)
│
│  ── Optimized Implementation (separate, do not mix) ──
├── phase5_graph_vectorized.py    Vectorized graph builder (21.6x faster)
├── phase5_algorithm2_pruned.py   Algorithm 2 with clique pruning + fast graph
├── phase6_bnb_optimized.py       B&B using pruned auxiliary (solve_bnb_optimized)
│
│  ── Validation Scripts ──
├── validate_gr202.py             Grotschel 202-city (Table 3)
├── validate_gr666.py             Grotschel 666-city (Table 4)
├── validate_body.py              Body measurements (Table 10)
├── validate_all.py               Unified runner
├── validate_optimized.py         Optimized vs original comparison
│
│  ── Datasets ──
├── iris.data                     Fisher's Iris (UCI, 150×4)
├── glass.data                    Glass Identification (UCI, 214×9)
├── gr202.tsp                     Grotschel 202-city (TSPLIB GEO format)
├── gr666.tsp                     Grotschel 666-city (TSPLIB GEO format)
├── body.dat.txt                  Body measurements (JSE, 507×25)
└── ruspini.csv                   Ruspini dataset (75×2)
```

---

## Quick Start

### Run a single instance

```python
import numpy as np
from phase1_foundation import MSSCInstance
from phase6_bnb import solve_bnb

# Load data
data = np.loadtxt("iris.data", delimiter=",", usecols=range(4))

# Create instance
inst = MSSCInstance(data, k=5, name="iris_k5")

# Solve
result = solve_bnb(inst, max_nodes=100, n_jmeans_restarts=5, seed=42, verbose=True)

print(f"Optimal cost : {result.optimal_cost:.4f}")
print(f"LP bound     : {result.lp_bound:.4f}")
print(f"Gap          : {result.gap_pct:.4f}%")
print(f"Status       : {result.status}")
print(f"Time         : {result.time_sec:.1f}s")
print(f"Nodes        : {result.n_nodes}")
print(f"Columns      : {result.n_columns}")
```

### Run with optimized solver

```python
from phase6_bnb_optimized import solve_bnb_optimized

result = solve_bnb_optimized(inst, max_nodes=100, n_jmeans_restarts=5,
                              seed=42, verbose=True)
```

### Run all benchmark validations

```bash
python validate_all.py iris
python validate_all.py glass
python validate_gr202.py
python validate_gr666.py
python validate_body.py
```

### Run optimized vs original comparison

```bash
python validate_optimized.py iris
python validate_optimized.py glass
python validate_optimized.py gr202
python validate_optimized.py gr666
```

---

## API Reference

### MSSCInstance

```python
from phase1_foundation import MSSCInstance

inst = MSSCInstance(
    points = np.ndarray,    # shape (n, s) — n points in s dimensions
    k      = int,           # number of clusters
    name   = str            # dataset name for logging
)

# Attributes
inst.n       # int — number of points
inst.s       # int — number of dimensions
inst.k       # int — number of clusters
inst.points  # np.ndarray shape (n, s)
inst.name    # str
```

### Cluster

```python
from phase1_foundation import Cluster, make_cluster

# Always use make_cluster — never construct Cluster directly
cl = make_cluster(inst, indices=tuple([0, 3, 7, 12]))

# Attributes
cl.indices   # tuple of int — member indices (MUST be tuple, not list)
cl.centroid  # np.ndarray shape (s,)
cl.cost      # float — within-cluster sum of squares
```

### RestrictedMasterProblem

```python
from phase3_rmp import RestrictedMasterProblem

rmp = RestrictedMasterProblem(inst)
rmp.add_column(cluster)           # add one column
rmp.add_columns(list_of_clusters) # add multiple columns

lp_obj, lam, sigma = rmp.solve()
# lp_obj : float      — LP objective value
# lam    : np.ndarray shape (n,) — covering duals (non-negative)
# sigma  : float      — cardinality dual (non-negative)

rc = rmp.reduced_cost(cluster, lam, sigma)
# rc < 0 means column can improve the LP

z = rmp.get_solution()  # np.ndarray — LP solution values
```

### solve_bnb

```python
from phase6_bnb import solve_bnb

result = solve_bnb(
    inst,
    max_nodes         = 200,   # B&B node limit
    n_jmeans_restarts = 5,     # j-means restarts for initial UB
    seed              = 42,    # RNG seed
    verbose           = True   # print B&B log
)

# BnBResult attributes
result.optimal_cost    # float — exact optimal MSSC cost
result.optimal_labels  # np.ndarray shape (n,) — cluster assignment
result.lp_bound        # float — LP lower bound at root
result.gap_pct         # float — optimality gap percentage
result.n_nodes         # int   — B&B nodes explored
result.n_columns       # int   — total columns generated
result.time_sec        # float — wall-clock time
result.status          # str   — 'optimal' or 'max_nodes'
```

---

## Loading Datasets

```python
import numpy as np

# Fisher's Iris (n=150, s=4)
iris = np.loadtxt("iris.data", delimiter=",", usecols=range(4))

# Glass Identification (n=214, s=9) — skip ID col 0 and class col 10
glass = np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10))

# Grotschel 202/666-city (n=202/666, s=2) — raw DDD.MM as 2D Euclidean
def parse_tsp(path):
    coords, in_section = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION": in_section = True; continue
            if line in ("EOF", "") and in_section: break
            if in_section:
                parts = line.split()
                coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)

gr202 = parse_tsp("gr202.tsp")
gr666 = parse_tsp("gr666.tsp")

# Body measurements (n=507, s=5) — use cols 9,10,11,22,23
body = np.loadtxt("body.dat.txt")[:, [9, 10, 11, 22, 23]]
```

---

## Critical Rules (Do Not Violate)

1. **Never add `if lp_obj >= ub - 1e-6: break` inside the CG loop** — this was Bug 1, caused premature termination without LP certificate
2. **Cluster indices must be a tuple** — use `make_cluster()`, never construct `Cluster()` directly with a list
3. **Do not negate Gurobi Pi for covering constraints** — `lam = np.maximum(lam, 0.0)` is correct; negation is wrong
4. **GEO coordinates for gr202/gr666** — use raw `DDD.MM` floats as 2D Euclidean; no geographic conversion
5. **Optimized files are separate** — `phase6_bnb_optimized.py` imports `phase5_graph_vectorized.py` and `phase5_algorithm2_pruned.py`; never import these into the original B&B

---

## Performance Expectations

| Dataset | n | s | k range | Time per k | Notes |
|---------|---|---|---------|------------|-------|
| Iris | 150 | 4 | 2–10 | 0.6–20s | Fast |
| Glass | 214 | 9 | 30–50 | 336–1987s | Slow due to high s |
| gr202 | 202 | 2 | 2–30 | 2–143s | Fast (2D aux) |
| gr666 | 666 | 2 | 2–50 | 17–4058s | Slow at large k |
| Body | 507 | 5 | 30–80 | 1300–9400s | Very slow |
| Telugu | 871 | 3 | any | timeout | Intractable in Python |

---

## Troubleshooting

**Gurobi license error**: ensure `grbgetkey` has been run and license file is at `~/gurobi.lic`

**`RMP solve failed with status X`**: initial columns don't cover all points — check `extract_initial_columns` returned valid clusters

**Cost much lower than paper**: wrong dataset columns — run `find_body_cols.py` pattern for body measurements

**Solver returns same cost as j-means instantly**: Bug 1 present — check CG loop has no early prune before auxiliary call

**`assert old in content` fails when patching files**: use `grep -n` to read exact whitespace before attempting string replacement
