# Exact MSSC Solver — Column Generation + Branch and Bound

An exact solver for the **Minimum Sum-of-Squares Clustering (MSSC)** problem, implementing and extending the column-generation Branch-and-Bound algorithm of Aloise, Hansen & Liberti (2012). The solver finds provably optimal k-means partitions via LP relaxation, an exact auxiliary subproblem, and Ryan–Foster branching. A vectorised multi-step screening heuristic (Phase 8) reduces initialisation time by up to 3–4× on high-dimensional instances and discovers solution quality improvements over the original paper's reported results.

## Based On

> D. Aloise, P. Hansen, L. Liberti, *"An improved column generation algorithm for minimum sum-of-squares clustering"*, **Mathematical Programming**, 134(2):539–565, 2012.  
> DOI: [10.1007/s10107-011-0437-7](https://doi.org/10.1007/s10107-011-0437-7)

## Features

- **Exact B&B solver** — certifies global optimality via LP lower bounds
- **Column generation** — dynamically adds optimal cluster columns to the RMP
- **Algorithm 1** — exact 2D auxiliary via disc-intersection geometry
- **Algorithm 2** — general-dimension auxiliary via hypersphere graph + Dinkelbach's algorithm
- **Ryan–Foster branching** — SAME/DIFF constraints on entity pairs
- **Pruned auxiliary** — singleton shortcut + bound pruning (40–70% fewer Dinkelbach calls)
- **Vectorised j-means** — adaptive multi-step candidate screening (Phase 8)
  - Fixes LP cycling failure mode present in original implementation
  - Finds better solutions than paper's reported Glass k=30 result
  - 1.8× overall speedup on high-dimensional instances
- Validated against all four paper benchmark datasets (Iris, Glass, gr202, gr666)

## Prerequisites

- Python 3.10+
- **Gurobi** (primary solver) — academic licence free at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/)
- numpy, scipy

**No Gurobi?** See [Alternative Solver](#alternative-solver) below.

## Installation

```bash
# 1. Clone
git clone https://github.com/deep-suman/mssc-bnb-solver.git
cd mssc-bnb-solver

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Gurobi licence setup
# Place gurobi.lic in your home directory, or set:
export GRB_LICENSE_FILE=/path/to/gurobi.lic

# 5. Verify
python -c "import gurobipy; print('Gurobi OK')"
python -c "import numpy; print('numpy', numpy.__version__)"
```

### Alternative Solver

The clean implementation (`mssc_clean.py`) includes a **PuLP fallback** for the RMP that works without a Gurobi licence:

```bash
pip install pulp
python mssc_clean.py --solver pulp --dataset iris --k 5
```

PuLP uses the CBC open-source solver. Results are identical; solve times are typically 2–5× slower than Gurobi.

## Project Structure

```
Implementation/
├── phase1_foundation.py          # Core data structures (MSSCInstance, Cluster, MSSCSolution)
├── phase2_kmeans.py              # k-means++ initialisation + Lloyd's algorithm
├── phase2_jmeans.py              # j-means local search (centroid teleportation)
├── phase3_rmp.py                 # Gurobi RMP (set-covering LP, barrier + crossover)
├── phase3_rmp_stabilised.py      # Stabilised RMP variant (in-out blending)
├── phase4_algorithm1.py          # Exact 2D auxiliary (disc-intersection geometry)
├── phase4_geometry.py            # Disc geometry primitives
├── phase4_regions.py             # Region enumeration for Algorithm 1
├── phase4_subproblem.py          # 2D subproblem solver
├── phase5_algorithm2.py          # General auxiliary (hypersphere graph + Dinkelbach)
├── phase5_graph.py               # Hypersphere intersection graph builder
├── phase5_graph_vectorized.py    # Vectorised graph builder (21.6× speedup)
├── phase5_dinkelbach.py          # Dinkelbach's algorithm for fractional QP
├── phase5_algorithm2_pruned.py   # Pruned auxiliary (singleton shortcut + bound pruning)
├── phase5_algorithm2_multi.py    # Multi-pricing variant (returns all negative-RC columns)
├── phase6_bnb.py                 # Original B&B solver
├── phase6_bnb_optimized.py       # Optimised B&B (pruned auxiliary)
├── phase6_branching.py           # Ryan–Foster branching logic
├── phase6_constrained_aux.py     # Constrained auxiliary for branching nodes
├── phase8_fast_jmeans.py         # Vectorised multi-step j-means (Phase 8 — main contribution)
├── phase8_bnb.py                 # Fast B&B using Phase 8 j-means
├── mssc_clean.py                 # Clean standalone implementation (dual Gurobi/PuLP)
├── validate_iris.py              # Iris benchmark (Table 8)
├── validate_glass.py             # Glass benchmark (Table 9)
├── validate_gr202.py             # gr202 benchmark (Table 3)
├── validate_gr666.py             # gr666 benchmark (Table 4)
├── validate_optimized.py         # Phase 6 optimised vs original comparison
├── validate_phase8.py            # Phase 8 fast vs original comparison
├── iris.data                     # Fisher's Iris dataset (n=150, s=4)
├── glass.data                    # Glass Identification dataset (n=214, s=9)
├── gr202.tsp                     # Grotschel 202-city TSP (n=202, s=2)
├── gr666.tsp                     # Grotschel 666-city TSP (n=666, s=2)
└── requirements.txt
```

## Usage

### Run the fast B&B solver on a single instance

```python
import numpy as np
from phase1_foundation import MSSCInstance
from phase8_bnb import solve_bnb_fast

points = np.loadtxt("iris.data", delimiter=",", usecols=range(4))
inst   = MSSCInstance(points, k=5, name="iris_k5")
result = solve_bnb_fast(inst, max_nodes=200, n_jmeans_restarts=5, seed=42, verbose=True)

print(f"Optimal cost : {result.optimal_cost:.4f}")
print(f"Gap          : {result.gap_pct:.4f}%")
print(f"Solve time   : {result.time_sec:.2f}s")
print(f"B&B nodes    : {result.n_nodes}")
```

Expected output:
```
  Fast B&B  |  iris_k5  |  n=150, k=5, s=4
  Initial UB = 46.5356
  Node   1 (depth=0): LP=46.5356  UB=46.5356  new_cols=18
         -> PRUNED
  Status       : optimal
  Optimal cost : 46.535600
  Gap          : 0.0000%
  Time         : 0.48s
```

### Run full dataset comparison (Phase 8 vs original)

```bash
python validate_phase8.py iris       # Iris k=2..10
python validate_phase8.py glass      # Glass k=30..50 (j-means only)
python validate_phase8.py gr202      # gr202 k=2..30
python validate_phase8.py gr666      # gr666 k=2..50
python validate_phase8.py all        # All datasets
```

### Run original paper validation

```bash
python validate_iris.py    # Paper Table 8
python validate_glass.py   # Paper Table 9
python validate_gr202.py   # Paper Table 3
python validate_gr666.py   # Paper Table 4
```

### Clean standalone implementation (with PuLP fallback)

```bash
# Gurobi (default)
python mssc_clean.py --dataset iris --k 5

# PuLP fallback (no licence needed)
python mssc_clean.py --dataset iris --k 5 --solver pulp

# Custom data
python mssc_clean.py --data my_data.csv --k 3 --seed 42

# Export results
python mssc_clean.py --dataset glass --k 30 --output results.json
```

## Configuration

Key parameters in `solve_bnb_fast()`:

| Parameter | Default | Description |
|---|---|---|
| `max_nodes` | 200 | Maximum B&B nodes before stopping |
| `n_jmeans_restarts` | 5 | Number of j-means random restarts |
| `seed` | 42 | Random seed for reproducibility |
| `verbose` | True | Print B&B progress |

Key parameters in `fast_jmeans()`:

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | None (auto) | Lloyd iterations per screening estimate (auto = 2 if n/k≥7 else 1) |
| `slack` | 0.05 | Fractional safety margin on screening threshold |

## Input Data Format

**CSV/whitespace-delimited numerical data** — one entity per row, one feature per column:

```
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
...
```

For TSP files, the standard `.tsp` NODE_COORD_SECTION format is parsed automatically.

Glass data uses columns 1–9 (skipping the ID column 0):
```python
points = np.loadtxt("glass.data", delimiter=",", usecols=range(1, 10))
```

## Output

`solve_bnb_fast()` returns a `BnBResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `optimal_cost` | float | Best integer solution cost (MSSC objective) |
| `optimal_labels` | np.ndarray | Cluster assignment for each entity (0-indexed) |
| `lp_bound` | float | LP lower bound at root node |
| `gap_pct` | float | Optimality gap in percent |
| `n_nodes` | int | Total B&B nodes explored |
| `n_columns` | int | Total columns added to RMP |
| `time_sec` | float | Wall-clock solve time |
| `status` | str | "optimal" or "max_nodes" |

## Troubleshooting

**Gurobi licence error:**
```
gurobipy.GurobiError: No Gurobi licence found
```
Download a free academic licence from gurobi.com/academia. Place `gurobi.lic` in your home directory.

**Memory error on large instances:**
Reduce `mem_limit_mb` in `_multi_step_costs()` (default 400 MB) or use `n_steps=1`.

**Solver returns `max_nodes` status:**
Increase `max_nodes`. The `optimal_cost` is a valid upper bound; `lp_bound` is the proven lower bound.

**Slow on large k (Glass k=45–50):**
Normal — Glass j-means takes 250–480s per run. Use Phase 8 solver which is 1.5–1.9× faster.

## License

Academic use only. Gurobi requires a separate licence (free for academia).
