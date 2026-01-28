# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

# -------------------------------------------------------------------------------------
# CHECK ATOMS NOT DESORBED
# -------------------------------------------------------------------------------------

def check_atoms_not_desorbed(
    atoms: Atoms,
    dist_ratio_thr: float = 1.40,
    **kwargs,
) -> bool:
    """
    Checks if the adsorbate atoms has not desorbed from the surface.
    """
    from arkimede.utilities import get_indices_adsorbate, get_connectivity
    # Get masks of adsorbate and slab atoms.
    mask_ads = get_indices_adsorbate(atoms=atoms, return_mask=True)
    mask_slab = [not aa for aa in mask_ads]
    # Get connectivity matrix.
    matrix = get_connectivity(atoms=atoms, dist_ratio_thr=dist_ratio_thr)
    # Return True if there is at least one bond.
    if np.sum(matrix[mask_ads][:, mask_slab]) > 0:
        return True
    else:
        return False

# -------------------------------------------------------------------------------------
# CHECK TS RELAX INTO IS AND FS
# -------------------------------------------------------------------------------------

def check_TS_relax_into_IS_and_FS(
    atoms_TS: Atoms,
    atoms_IS: Atoms,
    atoms_FS: Atoms,
    calc: Calculator,
    method: str = "vibrations",
    indices_check: list = "adsorbate",
    store_positions_relaxed: bool = False,
    **kwargs,
) -> bool:
    """
    Relax the TS atoms in two direction and check if the new structures have the
    same connectivity as the original IS and FS atoms.
    """
    from arkimede.workflow.calculations import run_calculation
    from arkimede.transtates import get_displaced_atoms_from_atoms_TS
    from arkimede.utilities import (
        get_indices_adsorbate,
        get_indices_not_fixed,
        check_same_connectivity,
    )
    # Get displaced atoms.
    if method == "irc":
        # IRC method.
        calculation = "irc"
        atoms_displaced = [atoms_TS.copy(), atoms_TS.copy()]
    else:
        # Relax displaced atoms.
        calculation = "relax"
        atoms_displaced = get_displaced_atoms_from_atoms_TS(
            atoms_TS=atoms_TS,
            method=method,
            calc=calc,
            **kwargs,
        )
    # Run calculations.
    atoms_new_list = []
    for atoms, direction in zip(atoms_displaced, ["forward", "reverse"]):
        run_calculation(
            atoms=atoms,
            calc=calc,
            calculation=calculation,
            direction=direction,
            **kwargs,
        )
        atoms_new_list.append(atoms)
    # Store positions of relaxed atoms in atoms info.
    if store_positions_relaxed is True:
        atoms_TS.info["positions_relaxed"] = [
            atoms_new_list[ii].positions for ii in (0, 1)
        ]
    # Get indices of atoms to check.
    if indices_check == "adsorbate":
        indices_check = get_indices_adsorbate(atoms=atoms)
    elif indices_check == "not_fixed":
        indices_check = get_indices_not_fixed(atoms=atoms)
    # Check connectivity matches.
    atoms_old_list = [atoms_IS, atoms_FS]
    for jj_list in [[0, 1], [1, 0]]:
        check_list = []
        for ii, jj in zip([0, 1], jj_list):
            check = check_same_connectivity(
                atoms_1=atoms_old_list[ii],
                atoms_2=atoms_new_list[jj],
                indices=indices_check,
            )
            check_list.append(check)
        # Return True if both connectivities are the same as the old ones.
        if np.all(check_list):
            return True
    # Return True if both connectivities are the same as the old ones.
    return False

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
