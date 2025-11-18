# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.optimize.climbfixinternals import (
    BFGSClimbFixInternals as BFGSClimbFixInternalsOriginal
)

# -------------------------------------------------------------------------------------
# BFGS CLIMB FIX INTERNALS
# -------------------------------------------------------------------------------------

class BFGSClimbFixInternals(BFGSClimbFixInternalsOriginal):
    def __init__(
        self,
        optB_max_steps: int = 1000,
        max_force_tot: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.optB_max_steps = optB_max_steps
        self.max_force_tot = max_force_tot
    
    def step(self):
        super().step()
        # Check maximum force.
        if self.max_force_tot is not None:
            forces_tot = self.get_projected_forces().reshape(-1, 3)
            if np.max(np.linalg.norm(forces_tot, axis=1)) > self.max_force_tot:
                raise RuntimeError("max force too high")

    def relax_remaining_dof(self):
        """
        Optimize remaining degrees of freedom with optimizer 'B'.
        """
        if self.autolog:
            self.optB_kwargs["logfile"] = f"optB_{self.targetvalue}.log"
        if self.autotraj:
            self.optB_kwargs["trajectory"] = f"optB_{self.targetvalue}.traj"
        fmax = self.get_scaled_fmax()
        with self.optB(self.optimizable.atoms, **self.optB_kwargs) as opt:
            opt.run(fmax, steps=self.optB_max_steps) # optimize with scaled fmax
            grad = self.optimizable.get_gradient()
            if self.converged(grad) and fmax > self.optB_fmax:
                # (final) optimization with desired fmax
                opt.run(self.optB_fmax)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
