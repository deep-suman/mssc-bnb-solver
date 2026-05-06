"""
Phase 8: Fast J-Means with Vectorised Multi-Step Candidate Screening
=====================================================================
Drop-in replacement for jmeans() in phase2_jmeans.py.

Core idea: instead of running full k-means convergence for every candidate
teleportation move, run n_steps Lloyd iterations batched across ALL m
candidates simultaneously using NumPy tensor ops.  Only candidates whose
estimated cost beats (1+slack) * current_best proceed to full k-means.

Key decisions
-------------
* n_steps=2 when n/k >= 7  (large clusters -- tight estimates needed)
* n_steps=1 otherwise      (small clusters -- 1 step already converges fast)
* s==2 fallback: for 2D instances Algorithm 1 dominates; vectorised
  screening adds overhead with no benefit, so fall back to original jmeans.
"""

import numpy as np
from typing import Optional, List
from phase1_foundation import MSSCInstance, MSSCSolution, Cluster, make_cluster
from phase2_kmeans import kmeans
from phase2_jmeans import extract_initial_columns, _try_jump


def _multi_step_costs(points, centroids, candidates, j, n_steps, mem_limit_mb=400):
    """
    Estimate MSSC cost for moving centroid j to each candidate entity,
    running n_steps Lloyd iterations batched across all m candidates.

    Returns (m,) cost array, or None if memory limit exceeded.
    """
    m = len(candidates)
    n, s = points.shape
    k = centroids.shape[0]

    if m * n * k * 8 > mem_limit_mb * 1024 * 1024:
        return None

    # Candidate tensor C[t] = centroids with row j -> points[candidates[t]]
    C = np.empty((m, k, s), dtype=np.float64)
    C[:] = centroids[np.newaxis, :, :]
    C[:, j, :] = points[candidates]

    for _ in range(n_steps):
        pts_sq = (points ** 2).sum(axis=1)           # (n,)
        C_sq   = (C ** 2).sum(axis=2)                # (m, k)
        cross  = np.einsum('is,mks->mik', points, C) # (m, n, k)
        dists  = pts_sq[np.newaxis, :, np.newaxis] - 2.0 * cross + C_sq[:, np.newaxis, :]
        labels = dists.argmin(axis=2)                # (m, n)

        one_hot = (labels[:, :, np.newaxis] == np.arange(k)[np.newaxis, np.newaxis, :]).astype(np.float64)
        counts  = one_hot.sum(axis=1)                # (m, k)
        sums    = np.einsum('mni,ns->mis', one_hot, points)
        denom   = np.maximum(counts, 1.0)[:, :, np.newaxis]
        new_C   = sums / denom
        C       = np.where((counts < 1)[:, :, np.newaxis], C, new_C)

    pts_sq = (points ** 2).sum(axis=1)
    C_sq   = (C ** 2).sum(axis=2)
    cross  = np.einsum('is,mks->mik', points, C)
    dists  = pts_sq[np.newaxis, :, np.newaxis] - 2.0 * cross + C_sq[:, np.newaxis, :]
    return dists.min(axis=2).sum(axis=1)             # (m,)


def _fast_jmeans_local_search(inst, sol, n_steps, slack):
    """J-means local search with vectorised multi-step candidate screening."""
    best     = sol
    improved = True

    while improved:
        improved  = False
        threshold = (1.0 + slack) * best.cost

        for j in range(inst.k):
            candidates = np.where(best.labels != j)[0]
            if len(candidates) == 0:
                continue

            est = _multi_step_costs(inst.points, best.centroids, candidates, j, n_steps)

            if est is None:
                screened = candidates
            else:
                mask = est < threshold
                if not mask.any():
                    continue
                prom_idx = np.where(mask)[0]
                order    = prom_idx[est[prom_idx].argsort()]
                screened = candidates[order]

            for i in screened:
                trial = _try_jump(inst, best.centroids, j, int(i))
                if trial is not None and trial.cost < best.cost - 1e-10:
                    best     = trial
                    improved = True
                    break

            if improved:
                break

    return best


def fast_jmeans(inst, init_solution=None, n_restarts=5, seed=0, n_steps=None, slack=0.05):
    """
    J-means heuristic using vectorised candidate screening.

    Drop-in replacement for jmeans() in phase2_jmeans.py.

    Adaptive n_steps: 2 when n/k >= 7 (large clusters need tighter estimates),
    else 1 (small clusters converge fast in one step).
    2D fallback: for s=2 reverts to original jmeans (Algorithm 1 dominates).
    """
    if inst.s == 2:
        from phase2_jmeans import jmeans
        return jmeans(inst, init_solution=init_solution,
                      n_restarts=n_restarts, seed=seed)

    if n_steps is None:
        n_steps = 2 if (inst.n / inst.k) >= 7.0 else 1

    rng = np.random.default_rng(seed)

    start = init_solution if init_solution is not None else kmeans(inst, seed=seed)
    best  = _fast_jmeans_local_search(inst, start, n_steps, slack)

    for _ in range(max(0, n_restarts - 1)):
        sol = kmeans(inst, seed=int(rng.integers(1_000_000)))
        sol = _fast_jmeans_local_search(inst, sol, n_steps, slack)
        if sol.cost < best.cost:
            best = sol

    return best
