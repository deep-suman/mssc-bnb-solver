"""
Phase 3 (stabilised): RMP with In-Out Blending (Stabilised Column Generation).

Stabilisation blends the current dual solution with a stability centre to
prevent oscillation in dual variables during column generation. This reduces
the number of CG iterations needed to solve the LP, especially for dense
instances where the standard barrier method oscillates.

Reference: Ben Amor et al. (2009), "Dual simplex stabilisation for column generation"
"""

import numpy as np
from typing import Optional, Tuple, List
from phase1_foundation import MSSCInstance, Cluster
from phase3_rmp import RestrictedMasterProblem


class StabilisedRMP:
    """
    Wrapper around RestrictedMasterProblem that applies in-out blending
    to stabilise dual variable trajectories.

    Parameters
    ----------
    inst        : MSSCInstance
    alpha       : float in (0, 1] — blending factor (1.0 = no stabilisation)
    """

    def __init__(self, inst: MSSCInstance, alpha: float = 0.5):
        self.inst = inst
        self.alpha = alpha
        self._rmp = RestrictedMasterProblem(inst)

        # Stability centre (initialised to zero)
        self._centre_lam: np.ndarray = np.zeros(inst.n)
        self._centre_sigma: float = 0.0
        self._n_cols: int = 0

    def add_column(self, col: Cluster) -> None:
        self._rmp.add_column(col)
        self._n_cols += 1

    def solve(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[float]]:
        """
        Solve stabilised RMP.

        Returns
        -------
        (obj, blended_lam, blended_sigma) or (None, None, None) on failure
        """
        obj, lam, sigma = self._rmp.solve()
        if obj is None or lam is None:
            return None, None, None

        # Blend with stability centre
        blended_lam = (self.alpha * lam
                       + (1.0 - self.alpha) * self._centre_lam)
        blended_sigma = (self.alpha * sigma
                         + (1.0 - self.alpha) * self._centre_sigma)

        # Update stability centre toward current solution
        self._centre_lam = blended_lam.copy()
        self._centre_sigma = float(blended_sigma)

        return obj, blended_lam, float(blended_sigma)

    def get_solution(self):
        return self._rmp.get_solution()

    @property
    def n_columns(self) -> int:
        return self._n_cols
