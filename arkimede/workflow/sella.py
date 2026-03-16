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
# HESSIAN FROM VIBRATIONS
# -------------------------------------------------------------------------------------

def hessian_from_vibrations(
    atoms: Atoms,
    delta: float = 0.01,
    label: str = "vib-uuid",
    indices: list = "not-fixed",
    remove_cache: bool = False,
):
    """
    Function to calculate the hessian matrix from finite elements.
    """
    from ase.vibrations import Vibrations
    from arkimede.utilities import get_indices_from_name
    # Get indices of atoms to vibrate.
    indices = get_indices_from_name(atoms=atoms, indices=indices)
    # Label with uuid to avoid conflicts.
    if label == "vib-uuid":
        label = f"vib-{uuid.uuid4().hex}"
    # Prepare vibrations calculation.
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    atoms.set_masses([1. for aa in atoms])
    # Run vibrations calculation.
    vib = Vibrations(atoms=atoms, indices=indices, delta=delta)
    vib.clean()
    vib.run()
    data = vib.get_vibrations()
    # Build full Hessian.
    cart_idx = np.repeat(3 * np.array(vib.indices), 3).reshape(-1, 3) + np.arange(3)
    hessian = np.zeros((3 * len(atoms), 3 * len(atoms)))
    hessian[np.ix_(cart_idx.ravel(), cart_idx.ravel())] = vib.H
    # Restore atoms object.
    atoms.set_positions(positions)
    atoms.set_masses(masses)
    # Remove the cache directory.
    if remove_cache is True:
        vib.clean()
        if os.path.isdir(label):
            shutil.rmtree(label)
    # Return Hessian.
    return hessian

# -------------------------------------------------------------------------------------
# HESSIAN FROM MODE TS
# -------------------------------------------------------------------------------------

def hessian_from_mode_TS(
    atoms: Atoms,
    scale: float = 1.0,
):
    """
    Function to initialize the hessian matrix with negative curvature in
    correspondance of the TS mode.
    """
    # Get data of mode TS.
    mode_TS = np.array(atoms.info["mode_TS"]).copy()
    # Build Hessian with negative curvature in correspondence of the TS mode.
    mode_TS = mode_TS.ravel()
    mode_TS /= np.linalg.norm(mode_TS)
    hessian = np.eye(len(mode_TS))
    hessian -= scale * np.outer(mode_TS, mode_TS)
    # Return Hessian.
    return hessian

# -------------------------------------------------------------------------------------
# HESSIAN FROM BONDS TS
# -------------------------------------------------------------------------------------

def hessian_from_bonds_TS(
    atoms: Atoms,
    sign_bond_dict: dict = {"break": +1, "form": -1},
    scale: float = 1.0,
):
    """
    Function to initialize the hessian matrix with negative curvature in
    correspondance of the TS bonds.
    """
    # Get data of TS bonds.
    bonds_TS = atoms.info["bonds_TS"]
    # Build Hessian with negative curvature in correspondence of TS bonds.
    mode_TS = np.zeros((len(atoms), 3))
    for bond in bonds_TS:
        index_a, index_b, sign = bond
        if isinstance(sign, str):
            sign = sign_bond_dict[sign]
        dir_bond = atoms.positions[index_a] - atoms.positions[index_b]
        mode_TS[index_a] += dir_bond * sign
        mode_TS[index_b] -= dir_bond * sign
    mode_TS = mode_TS.ravel()
    mode_TS /= np.linalg.norm(mode_TS)
    hessian = np.eye(len(mode_TS))
    hessian -= scale * np.outer(mode_TS, mode_TS)
    # Return Hessian.
    return hessian

# -------------------------------------------------------------------------------------
# MODIFY HESSIAN OBS
# -------------------------------------------------------------------------------------

def modify_hessian_obs(
    opt: Optimizer,
    bonds_TS: list,
    delta: float = 0.1,
    dot_prod_thr: float = 0.5,
    sign_bond_dict: dict = {"break": +1, "form": -1},
):
    """
    Observer that modifies the hessian matrix and set negative curvature in
    corrispondance of the TS bonds.
    """
    # Get Hessian.
    hessian = opt.pes.H.B
    if hessian is None or bonds_TS is None:
        return
    # Get bonds vector.
    vector = np.zeros((len(opt.atoms), 3))
    for bond in bonds_TS:
        index_a, index_b, sign = bond
        if isinstance(sign, str):
            sign = sign_bond_dict[sign]
        dir_bond = opt.atoms.positions[index_a] - opt.atoms.positions[index_b]
        dir_bond /= np.linalg.norm(dir_bond)
        vector[index_a] += dir_bond * sign
        vector[index_b] -= dir_bond * sign
    vector = vector.ravel()
    vector /= np.linalg.norm(vector)
    # Get eigenvector.    
    evec = np.linalg.eigh(hessian)[1][:, 0]
    dot_prod = np.abs(np.dot(evec, vector))
    while dot_prod < dot_prod_thr:
        hessian = hessian - delta * np.outer(vector, vector)
        evec = np.linalg.eigh(hessian)[1][:, 0]
        dot_prod = np.abs(np.dot(evec, vector))
    # Update Hessian.
    opt.pes.set_H(target=hessian)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
