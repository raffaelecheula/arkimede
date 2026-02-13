# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

# -------------------------------------------------------------------------------------
# MIN STEPS OBS
# -------------------------------------------------------------------------------------

def min_steps_obs(
    opt: Optimizer,
    min_steps: int,
    fmax: float,
):
    """
    Observer setting a minimum number of steps.
    """
    if opt.nsteps >= min_steps:
        opt.fmax = fmax
    else:
        opt.fmax = 0.

# -------------------------------------------------------------------------------------
# MAX FORCECALLS OBS
# -------------------------------------------------------------------------------------

def max_forcecalls_obs(
    opt: Optimizer,
    max_forcecalls: int,
    calc: Calculator,
):
    """
    Observer setting a maximum number of forces calls.
    """
    if "counter" in dir(calc) and calc.counter > max_forcecalls:
        opt.max_steps = 0

# -------------------------------------------------------------------------------------
# RESET EIGENMODE OBS
# -------------------------------------------------------------------------------------

def reset_eigenmode_obs(
    atoms: Atoms,
    bonds_TS: list,
    sign_bond_dict: dict = {"break": +1, "form": -1},
):
    """
    Observer resetting the eigenmode of a dimer calculation.
    """
    from arkimede.transtates import get_mode_TS_from_bonds_TS
    mode_TS = get_mode_TS_from_bonds_TS(
        atoms=atoms,
        bonds_TS=bonds_TS,
        sign_bond_dict=sign_bond_dict,
    )
    atoms.eigenmodes = [mode_TS]

# -------------------------------------------------------------------------------------
# MAX DISPLACEMENT OBS
# -------------------------------------------------------------------------------------

def max_displacement_obs(
    atoms: Atoms,
    atoms_zero: Atoms,
    max_displ: float,
):
    """
    Observer stopping the calculation when the atoms are too distant from the start.
    """
    displ = np.linalg.norm(atoms.positions - atoms_zero.positions)
    if displ > max_displ:
        raise RuntimeError("atoms displaced too much")

# -------------------------------------------------------------------------------------
# MAX FORCE TOT OBS
# -------------------------------------------------------------------------------------

def max_force_tot_obs(
    atoms: Atoms,
    max_force_tot: float,
):
    """
    Observer stopping the calculation when the maximum force is too high.
    """
    if np.max(np.linalg.norm(atoms.get_forces(), axis=1)) > max_force_tot:
        raise RuntimeError("max force too high")

# -------------------------------------------------------------------------------------
# ATOMS TOO CLOSE OBS
# -------------------------------------------------------------------------------------

def atoms_too_close_obs(
    atoms: Atoms,
    mindist: float = 0.2,
):
    """
    Observer stopping the calculation when two atoms are too close to each other.
    """
    for ii, position in enumerate(atoms.positions):
        distances = np.linalg.norm(atoms.positions[ii+1:] - position, axis=1)
        duplicates = np.where(distances < mindist)[0]
        if len(duplicates) > 0:
            raise RuntimeError("atoms too close")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
