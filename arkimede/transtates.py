# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import LBFGS
try: from ase.mep.neb import NEB, idpp_interpolate
except: from ase.neb import NEB, idpp_interpolate
from ase.calculators.singlepoint import SinglePointCalculator, all_properties

# -------------------------------------------------------------------------------------
# GET IDPP INTERPOLATED IMAGES
# -------------------------------------------------------------------------------------

def get_idpp_interpolated_images(
    atoms_IS: Atoms,
    atoms_FS: Atoms,
    n_images: int = 10,
    label: str = "idpp",
    fmax: float = 0.01,
    traj: str = None,
    log: str = None, 
    optimizer: Optimizer = LBFGS,
    mic: bool = False,
    steps_max: int = 1000,
    save_trajs: bool = False,
    from_atoms_info: bool = False,
) -> list:
    """
    Get idpp interpolated images with strict optimization parameters.
    """
    from ase.io import Trajectory
    images = [atoms_IS.copy()]
    images += [atoms_IS.copy() for i in range(n_images-2)]
    images += [atoms_FS.copy()]
    neb = NEB(images=images)
    neb.interpolate()
    idpp_interpolate(
        images=images,
        traj=traj,
        log=log, 
        fmax=fmax,
        optimizer=optimizer,
        mic=mic,
        steps=steps_max,
    )
    if save_trajs is True:
        traj = Trajectory(filename=f"{label}.traj", mode="w")
        for image in images:
            traj.write(image)
    # Return images.
    return images

# -------------------------------------------------------------------------------------
# GET ATOMS TS FROM IMAGES NEB
# -------------------------------------------------------------------------------------

def get_atoms_TS_from_images_neb(
    images: list,
    atoms_TS: Atoms = None,
    index_TS: int = None,
    no_IS_or_FS: bool = False,
    store_images_data: bool = False,
    calc: Calculator = None,
    properties: list = all_properties,
    update_cell: bool = False,
) -> Atoms:
    """
    Get the atoms of the TS guess and the approximate TS mode from the NEB images.
    """
    from arkimede.utilities import update_atoms_from_atoms_opt
    # Get images energies.
    images_energies = []
    for atoms in images:
        if calc is not None:
            atoms.calc = calc
            atoms.info["status"] = "unfinished"
        images_energies.append(atoms.get_potential_energy())
    # Get index of transition state.
    if index_TS is None:
        index_TS = np.argmax(images_energies)
    # Avoid returning a initial or final state.
    if no_IS_or_FS is True:
        if index_TS == 0:
            index_TS = 1
        elif index_TS == len(images) - 1:
            index_TS = len(images) - 2 
    # Select TS atoms or update given atoms.
    if atoms_TS is None:
        atoms_TS = images[index_TS].copy()
    else:
        update_atoms_from_atoms_opt(
            atoms=atoms_TS,
            atoms_opt=images[index_TS],
            status=images[index_TS].info["status"],
            properties=properties,
            update_cell=update_cell,
        )
    # Store data of images in atoms info dictionary.
    if store_images_data is True:
        images_positions = np.vstack([atoms.positions.copy() for atoms in images])
        atoms_TS.info.update({
            "images_positions": images_positions,
            "images_energies": images_energies,
        })
    # Get the approximate TS mode.
    mode_TS = get_mode_TS_from_images(images=images, index_TS=index_TS)
    atoms_TS.info["mode_TS"] = mode_TS
    # Return TS atoms.
    return atoms_TS

# -------------------------------------------------------------------------------------
# GET MODE TS FROM IMAGES
# -------------------------------------------------------------------------------------

def get_mode_TS_from_images(
    images: list,
    index_TS: int,
) -> np.ndarray:
    """
    Get TS mode from images of a NEB calculation.
    """
    mode_TS = np.zeros((len(images[0]), 3))
    if index_TS > 0:
        mode_TS += images[index_TS].positions - images[index_TS-1].positions
    if index_TS < len(images) - 1:
        mode_TS += images[index_TS+1].positions - images[index_TS].positions
    mode_TS /= np.linalg.norm(mode_TS)
    # Return approximate TS mode.
    return mode_TS

# -------------------------------------------------------------------------------------
# GET MODE TS FROM BONDS TS
# -------------------------------------------------------------------------------------

def get_mode_TS_from_bonds_TS(
    atoms: Atoms,
    bonds_TS: list = None,
    sign_bond_dict: dict = {"break": +1, "form": -1},
    **kwargs,
) -> np.ndarray:
    """
    Get TS mode from the TS bonds.
    """
    if bonds_TS is None:
        bonds_TS = atoms.info["bonds_TS"]
    mode_TS = np.zeros((len(atoms), 3))
    for bond in bonds_TS:
        index_a, index_b, sign_bond = bond
        if isinstance(sign_bond, str):
            sign_bond = sign_bond_dict[sign_bond]
        dir_bond = atoms.positions[index_a] - atoms.positions[index_b]
        mode_TS[index_a] += dir_bond * sign_bond
        mode_TS[index_b] -= dir_bond * sign_bond
    mode_TS /= np.linalg.norm(mode_TS)
    # Return TS mode.
    return mode_TS

# -------------------------------------------------------------------------------------
# GET MODE TS FROM VIBRATIONS
# -------------------------------------------------------------------------------------

def get_mode_TS_from_vibrations(
    atoms: Atoms,
    calc: Calculator,
    delta: float = 0.01,
    indices: list = "not_fixed",
    label: str = "vib",
    clean_cache: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Get vibrational energies and modes.
    """
    import shutil
    from ase.vibrations import Vibrations
    from arkimede.utilities import get_indices_adsorbate, get_indices_not_fixed
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    # Set indices not fixed if not provided.
    if indices == "adsorbate":
        indices = get_indices_adsorbate(atoms=atoms)
    elif indices == "not_fixed":
        indices = get_indices_not_fixed(atoms=atoms)
    # Run vibrations calculations.
    vib = Vibrations(atoms=atoms_copy, indices=indices, delta=delta, name=label)
    vib.clean()
    vib.run()
    data = vib.get_vibrations()
    energies, modes = data.get_energies_and_modes(all_atoms=True)
    if clean_cache is True:
        vib.clean()
        if os.path.isdir(label):
            shutil.rmtree(label)
    # Return energies and modes.
    return modes[0]

# -------------------------------------------------------------------------------------
# ATOMS IMAGES NEB FROM ATOMS TS 
# -------------------------------------------------------------------------------------

def get_images_neb_from_atoms_TS(
    atoms_TS: Atoms,
) -> list:
    """
    Get images from atoms TS (with images positions and energies in info).
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    images_energies = atoms_TS.info["images_energies"]
    images_positions = np.split(
        atoms_TS.info["images_positions"], len(images_energies)
    )
    images = [atoms_TS.copy() for positions in images_positions]
    for ii, positions in enumerate(images_positions):
        images[ii].set_positions(positions)
        images[ii].calc = SinglePointCalculator(
            atoms=images[ii],
            energy=images_energies[ii],
        )
    # Return images.
    return images

# -------------------------------------------------------------------------------------
# GET NEW IS AND FS FROM IMAGES
# -------------------------------------------------------------------------------------

def get_new_IS_and_FS_from_images(
    images: list,
) -> list:
    """
    Get new initial and final states (to be relaxed) from NEB images.
    """
    index_TS = np.argmax([image.get_potential_energy() for image in images])
    index_IS = index_TS-1 if index_TS > 0 else index_TS
    index_FS = index_TS+1 if index_TS < len(images) else index_TS
    atoms_IS_new = images[index_IS].copy()
    atoms_FS_new = images[index_FS].copy()
    # Return new initial and final states.
    return atoms_IS_new, atoms_FS_new

# -------------------------------------------------------------------------------------
# GET DISPLACED ATOMS FROM ATOMS TS
# -------------------------------------------------------------------------------------

def get_displaced_atoms_from_atoms_TS(
    atoms_TS: Atoms,
    method: str,
    calc: Calculator = None,
    mode_TS: np.ndarray = None,
    displ_tot: float = 0.20,
    displ_max: float = 0.10,
    **kwargs: dict,
) -> list:
    """
    Get displaced atoms from TS atoms (for calculating new IS and FS).
    """
    if method not in ("mode_TS", "bonds_TS", "vibrations"):
        raise Exception("method should be mode_TS, bonds_TS, or vibrations")
    # Calculate TS mode.
    if method == "bonds_TS":
        mode_TS = get_mode_TS_from_bonds_TS(atoms=atoms_TS, **kwargs)
    elif method == "vibrations":
        mode_TS = get_mode_TS_from_vibrations(atoms=atoms_TS, calc=calc, **kwargs)
    # Set displacement magnitude.
    mode_TS *= displ_tot / np.linalg.norm(mode_TS)
    if np.max(np.linalg.norm(mode_TS, axis=1)) > displ_max:
        mode_TS *= displ_max / np.max(np.linalg.norm(mode_TS, axis=1))
    # Apply displacement with positive and negative signs.
    atoms_displaced = [atoms_TS.copy(), atoms_TS.copy()]
    for sign, atoms in zip([-1, +1], atoms_displaced):
        atoms.positions += sign * mode_TS
    # Return list of displaced atoms.
    return atoms_displaced

# -------------------------------------------------------------------------------------
# GET ACTIVATION ENERGIES
# -------------------------------------------------------------------------------------

def get_activation_energies(
    images: list,
) -> list:
    """
    Calculate activation energies from images.
    """
    energy_TS = max([atoms.get_potential_energy() for atoms in images])
    eact_for = energy_TS - images[0].get_potential_energy()
    eact_rev = energy_TS - images[-1].get_potential_energy()
    return eact_for, eact_rev

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
