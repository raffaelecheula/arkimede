# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
try: from ase.mep.neb import NEB
except: from ase.neb import NEB

# -------------------------------------------------------------------------------------
# ACTIVATE CLIMB OBS
# -------------------------------------------------------------------------------------

def activate_climb_obs(
    neb: NEB,
    ftres: float,
    use_fmax_TS: bool = True,
) -> None:
    """
    Observer activating the climbing image method in a NEB calculation.
    """
    if use_fmax_TS is True:
        fmax = np.linalg.norm(neb.images[neb.imax].get_forces(), axis=1).max()
    else:
        fmax = neb.get_residual()
    if fmax < ftres:
        neb.climb = True

# -------------------------------------------------------------------------------------
# PRINT ENERGIES OBS
# -------------------------------------------------------------------------------------

def print_energies_obs(
    neb: NEB,
    opt: Optimizer,
    print_path_energies: bool = False,
):
    """
    Observer printing the energies of the images of a NEB calculation.
    """
    # Print step, neb residuals, and TS max force.
    fres = neb.get_residual()
    fmax_TS = np.linalg.norm(neb.images[neb.imax].get_forces(), axis=1).max()
    for ii in (0, -1):
        neb.energies[ii] = neb.images[ii].calc.results["energy"]
    act_energy = max(neb.energies) - neb.energies[0]
    print(f"Step: {opt.nsteps:4d}", end=" | ")
    print(f"Fres: {fres:6.3f} [eV/Å]", end=" | ")
    print(f"FmaxTS: {fmax_TS:6.3f} [eV/Å]", end=" | ")
    print(f"Eact: {act_energy:+7.3f} [eV]", end=" | ")
    # Print path energies.
    if print_path_energies is True:
        print(f"Epath:", end=" ")
        for energy in neb.energies:
            print(f"{energy-neb.energies[0]:+7.3f}", end=" ")
        print("[eV]")

# -------------------------------------------------------------------------------------
# CONVERGED TS OBS
# -------------------------------------------------------------------------------------

def converged_TS_obs(
    neb: NEB,
    opt: Optimizer,
    fmax: float,
):
    """
    Observer activating the climbing image method in a NEB calculation.
    """
    fmax_TS = np.linalg.norm(neb.images[neb.imax].get_forces(), axis=1).max()
    if neb.climb is True and fmax_TS < fmax:
        raise TSConverged

# -------------------------------------------------------------------------------------
# TS CONVERGED
# -------------------------------------------------------------------------------------

class TSConverged(Exception):
    """
    Raised when the transition state in the NEB calculation has converged.
    """
    pass

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
