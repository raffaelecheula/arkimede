# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import uuid
import shutil
import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

# -------------------------------------------------------------------------------------
# GET HESSIAN FROM VIBRATIONS FUN
# -------------------------------------------------------------------------------------

def get_hessian_from_vibrations_fun(
    indices_vib: list = "adsorbate",
    label: str = "hessian",
    **kwargs: dict,
):
    """
    Get a function to calculate the Hessian matrix from finite elements.
    """
    from arkimede.utilities import get_indices_from_name
    from arkimede.workflow.calculations import run_vibrations_calculation
    # Function to calculate the Hessian matrix.
    def hessian_from_vibrations(atoms: Atoms):
        # Get indices of atoms to vibrate.
        indices_vib_ii = get_indices_from_name(atoms=atoms, indices=indices_vib)
        # Run vibrations calculations.
        vib_hessian, vib_energies, vib_modes = run_vibrations_calculation(
            atoms=atoms,
            calc=atoms.calc,
            indices_vib=indices_vib_ii,
            label=label,
            unit_masses=True,
            **kwargs,
        )
        # Build full Hessian.
        idx = np.repeat(3 * np.array(indices_vib_ii), 3).reshape(-1, 3) + np.arange(3)
        hessian = np.zeros((3 * len(atoms), 3 * len(atoms)))
        hessian[np.ix_(idx.ravel(), idx.ravel())] = vib_hessian
        # Return Hessian.
        return hessian
    # Return function.
    return hessian_from_vibrations

# -------------------------------------------------------------------------------------
# GET HESSIAN FROM MODE TS FUN
# -------------------------------------------------------------------------------------

def get_hessian_from_mode_TS_fun(
    scale: float = 1.0,
    **kwargs: dict,
):
    """
    Get a function to initialize the Hessian matrix with negative curvature in
    correspondance of the TS mode.
    """
    def hessian_from_mode_TS(atoms: Atoms):
        # Get data of mode TS.
        mode_TS = np.array(atoms.info["mode_TS"]).copy()
        # Build Hessian with negative curvature in correspondence of the TS mode.
        mode_TS = mode_TS.ravel()
        mode_TS /= np.linalg.norm(mode_TS)
        hessian = np.eye(len(mode_TS))
        hessian -= scale * np.outer(mode_TS, mode_TS)
        # Return Hessian.
        return hessian
    # Return function.
    return hessian_from_mode_TS

# -------------------------------------------------------------------------------------
# GET HESSIAN FROM BONDS TS
# -------------------------------------------------------------------------------------

def get_hessian_from_bonds_TS_fun(
    sign_bond_dict: dict = {"break": +1, "form": -1},
    scale: float = 1.0,
    **kwargs: dict,
):
    """
    Get a function to initialize the Hessian matrix with negative curvature in
    correspondance of the TS bonds.
    """
    def hessian_from_bonds_TS(atoms: Atoms):
        # Get data of TS bonds.
        bonds_TS = atoms.info["bonds_TS"]
        # Build Hessian with negative curvature in correspondence of TS bonds.
        mode_TS = np.zeros((len(atoms), 3))
        for bond in bonds_TS:
            index_a, index_b, sign = bond
            if isinstance(sign, str):
                sign = sign_bond_dict[sign]
            dir_bond = atoms.get_distance(
                a0=index_b,
                a1=index_a,
                mic=True,
                vector=True,
            )
            dir_bond /= np.linalg.norm(dir_bond)
            mode_TS[index_a] += dir_bond * sign
            mode_TS[index_b] -= dir_bond * sign
        mode_TS = mode_TS.ravel()
        mode_TS /= np.linalg.norm(mode_TS)
        hessian = np.eye(len(mode_TS))
        hessian -= scale * np.outer(mode_TS, mode_TS)
        # Return Hessian.
        return hessian
    # Return function.
    return hessian_from_bonds_TS

# -------------------------------------------------------------------------------------
# GET HESSIAN FROM INPUT FUN
# -------------------------------------------------------------------------------------

def get_hessian_from_input_fun(
    **kwargs: dict,
):
    """
    Get a function to initialize the Hessian from the input Hessian.
    """
    def hessian_from_input(atoms: Atoms):
        # Return input Hessian.
        return np.array(atoms.info["hessian"]).copy()
    # Return function.
    return hessian_from_input

# -------------------------------------------------------------------------------------
# MODIFY HESSIAN OBS
# -------------------------------------------------------------------------------------

def modify_hessian_obs(
    opt: Optimizer,
    bonds_TS: list,
    delta: float = 1.0,
    dot_prod_thr: float = 0.5,
    iter_max: int = 10000,
    smooth_thr: bool = True,
    sign_bond_dict: dict = {"break": +1, "form": -1},
):
    """
    Observer that modifies the Hessian matrix and set negative curvature in
    correspondance of the TS bonds.
    """
    # Get Hessian.
    hessian = opt.pes.H.B
    if hessian is None or bonds_TS is None or dot_prod_thr == 0.:
        return
    # Get bonds vector.
    if opt.user_internal is not False:
        vector = np.zeros(hessian.shape[0])
        for ii, bond in enumerate(bonds_TS):
            index_a, index_b, sign = bond
            if isinstance(sign, str):
                sign = sign_bond_dict[sign]
            vector[ii] = sign
    else:
        vector = np.zeros((len(opt.atoms), 3))
        for bond in bonds_TS:
            index_a, index_b, sign = bond
            if isinstance(sign, str):
                sign = sign_bond_dict[sign]
            dir_bond = opt.atoms.get_distance(
                a0=index_b,
                a1=index_a,
                mic=True,
                vector=True,
            )
            dir_bond /= np.linalg.norm(dir_bond)
            vector[index_a] += dir_bond * sign
            vector[index_b] -= dir_bond * sign
    vector = vector.ravel()
    vector /= np.linalg.norm(vector)
    # Get eigenvector.
    evec = np.linalg.eigh(hessian)[1][:, 0]
    dot_prod = np.abs(np.dot(evec, vector))
    # Get smoothed dot product threshold.
    if smooth_thr is True:
        p_exp = 2 + 1 / (1 - dot_prod_thr) ** 2
        dot_prod_thr = (dot_prod ** p_exp + dot_prod_thr ** p_exp) ** (1 / p_exp)
    # Modify Hessian.
    outer = np.outer(vector, vector)
    for ii in range(iter_max):
        if dot_prod >= dot_prod_thr:
            break
        hessian = hessian - delta * outer
        evec = np.linalg.eigh(hessian)[1][:, 0]
        dot_prod = np.abs(np.dot(evec, vector))
    # Update Hessian.
    opt.pes.set_H(target=hessian)

# -------------------------------------------------------------------------------------
# GET INTERNALS
# -------------------------------------------------------------------------------------

def get_internals(
    atoms: Atoms,
    dist_ratio_thr: float = 1.50,
    dist_ratio_thr_ads: float = None,
    find_angles: bool = False,
    find_dihedrals: bool = False,
):
    """
    Find the key internal coordinates from reaction path.
    """
    from sella import Internals, Constraints
    from arkimede.utilities import get_edges_list, get_indices_fixed
    # Get internals object.
    internals = Internals(
        atoms=atoms,
        cons=Constraints(atoms=atoms),
        allow_fragments=False,
    )
    # Get internal.
    edges = get_edges_list(atoms=atoms, dist_ratio_thr=dist_ratio_thr)
    fixed = get_indices_fixed(atoms=atoms)
    bonds = [bb[:2] for bb in atoms.info["bonds_TS"]]
    for ee in edges:
        if (ee[0] not in fixed and ee[1] not in fixed) and ee not in bonds:
            bonds.append(ee)
    # Get adsorbate bonds.
    if dist_ratio_thr_ads is not None:
        indices_ads = atoms.info["indices_ads"]
        edges_ads = get_edges_list(atoms=atoms, dist_ratio_thr=dist_ratio_thr_ads)
        for ee in edges_ads:
            if (ee[0] in indices_ads or ee[1] in indices_ads) and ee not in bonds:
                bonds.append(ee)
    # Add bonds to internals.
    for bond in bonds:
        internals.add_bond(bond)
    # Add angles to internals.
    if find_angles is True:
        internals.find_all_angles()
    # Add dihedrals to internals.
    if find_dihedrals is True:
        internals.find_all_dihedrals()
    # Return internals.
    return internals

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
