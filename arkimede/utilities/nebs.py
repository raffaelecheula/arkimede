# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

# -------------------------------------------------------------------------------------
# GET IDPP INTERPOLATED IMAGES
# -------------------------------------------------------------------------------------

def get_idpp_interpolated_images(
    atoms_IS,
    atoms_FS,
    n_images=10,
    label='idpp',
    fmax=0.01,
    steps_max=1000,
    save_trajs=False,
):
    """Get idpp interpolated images with strict optimization parameters."""
    from ase.neb import NEB, idpp_interpolate
    from ase.optimize import LBFGS
    from ase.io import Trajectory
    images = [atoms_IS.copy()]
    images += [atoms_IS.copy() for i in range(n_images-2)]
    images += [atoms_FS.copy()]
    neb = NEB(images=images)
    neb.interpolate()
    idpp_interpolate(
        images=images,
        traj=None,
        log=None, 
        fmax=fmax,
        optimizer=LBFGS,
        mic=False,
        steps=steps_max,
    )
    if save_trajs is True:
        traj = Trajectory(filename=f'{label}.traj', mode='w')
        for image in images:
            traj.write(image)

    return images

# -------------------------------------------------------------------------------------
# GET ATOMS TS FROM IMAGES NEB
# -------------------------------------------------------------------------------------

def get_atoms_TS_from_images_neb(images, atoms_TS=None, index_TS=None):
    """Get the atoms of the TS guess and the vector for dimer calculation 
    from the NEB images."""

    images_energies = [atoms.get_potential_energy() for atoms in images]
    if index_TS is None:
        index_TS = np.argmax(images_energies)

    if atoms_TS is None:
        atoms_TS = images[index_TS].copy()
    else:
        atoms_TS.set_positions(images[index_TS].positions)
        atoms_TS.info["modified"] = images[index_TS].info["modified"]
        atoms_TS.info["converged"] = images[index_TS].info["converged"]
        atoms_TS.calc = SinglePointCalculator(
            atoms=atoms_TS,
            energy=images[index_TS].get_potential_energy(),
            forces=images[index_TS].get_forces(),
        )
    
    atoms_TS.info.update({
        "images_positions": np.vstack([atoms.positions.copy() for atoms in images]),
        "images_energies": [atoms.get_potential_energy() for atoms in images],
    })
    
    vector = get_vector_from_images(images, index_TS)
    atoms_TS.info["vector"] = vector
    
    return atoms_TS

# -------------------------------------------------------------------------------------
# FIX ATOMS FS MIC
# -------------------------------------------------------------------------------------

def fix_atoms_FS_mic(atoms_IS, atoms_FS):
    """Fix atoms positions of FS according to the minimum image convention."""
    from ase.geometry import get_distances
    for ii in range(len(atoms_IS)):
        displ, _ = get_distances(
            p1=atoms_IS[ii].position,
            p2=atoms_FS[ii].position,
            cell=atoms_IS.cell,
            pbc=True,
        )
        displ_tot = displ[0] if ii == 0 else np.vstack([displ_tot, displ[0]])
    atoms_FS.positions = atoms_IS.positions+displ_tot

# -------------------------------------------------------------------------------------
# GET VECTOR FROM IMAGES
# -------------------------------------------------------------------------------------

def get_vector_from_images(images, index_TS):
    """Get vector for a dimer calculation from images of a NEB calculation."""
    vector = np.zeros((len(images[0]), 3))
    if index_TS > 0:
        vector += images[index_TS].positions-images[index_TS-1].positions
    if index_TS < len(images)-1:
        vector += images[index_TS+1].positions-images[index_TS].positions
    vector /= np.linalg.norm(vector)
    
    return vector

# -------------------------------------------------------------------------------------
# GET VECTOR FROM IMAGES
# -------------------------------------------------------------------------------------

def get_vector_from_bonds_TS(
    atoms,
    bonds_TS,
    sign_bond_dict={'break': +1, 'form': -1},
):
    """Get vector for a dimer calculation from the TS bonds."""
    vector = np.zeros((len(atoms), 3))
    for bond in bonds_TS:
        index_a, index_b, sign_bond = bond
        if isinstance(sign_bond, str):
            sign_bond = sign_bond_dict[sign_bond]
        dir_bond = (
            atoms.positions[index_a]-atoms.positions[index_b]
        )
        vector[index_a] += +dir_bond * sign_bond
        vector[index_b] += -dir_bond * sign_bond
    vector /= np.linalg.norm(vector)
    
    return vector

# -------------------------------------------------------------------------------------
# ATOMS IMAGES NEB FROM ATOMS TS 
# -------------------------------------------------------------------------------------

def get_images_neb_from_atoms_TS(atoms_TS):
    """Get images from atoms TS (with images positions and energies in info)."""
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
    
    return images

# -------------------------------------------------------------------------------------
# GET NEW IS AND FS FROM IMAGES
# -------------------------------------------------------------------------------------

def get_new_IS_and_FS_from_images(images):
    """Get new initial and final states (to be relaxed) from NEB images."""
    index_TS = np.argmax([image.get_potential_energy() for image in images])
    index_IS = index_TS-1 if index_TS > 0 else index_TS
    index_FS = index_TS+1 if index_TS < len(images) else index_TS
    atoms_IS_new = images[index_IS].copy()
    atoms_FS_new = images[index_FS].copy()

    return atoms_IS_new, atoms_FS_new

# -------------------------------------------------------------------------------------
# GET NEW IS AND FS FROM ATOMS TS
# -------------------------------------------------------------------------------------

def get_new_IS_and_FS_from_atoms_TS(
    atoms_TS=None,
    vector=None,
    bonds_TS=None,
    delta=0.05,
    sign_bond_dict={'break': +1, 'form': -1},
):
    """Get new initial and final states (to be relaxed) from TS atoms."""
    atoms_IS_new = atoms_TS.copy()
    atoms_FS_new = atoms_TS.copy()
    signs_images = [-1, +1]
    if vector is not None:
        for ii, atoms in enumerate([atoms_IS_new, atoms_FS_new]):
            atoms.positions += signs_images[ii] * vector * delta
    elif bonds_TS is not None:
        for bond in bonds_TS:
            index_a, index_b, sign_bond = bond
            if isinstance(sign_bond, str):
                sign_bond = sign_bond_dict[sign_bond]
            dir_bond = atoms_TS.positions[index_a]-atoms_TS.positions[index_b]
            for ii, atoms in enumerate([atoms_IS_new, atoms_FS_new]):
                sign_tot = sign_bond * signs_images[ii]
                atoms.positions[index_a] += sign_tot * dir_bond * delta
                atoms.positions[index_b] -= sign_tot * dir_bond * delta
    else:
        raise RuntimeError("vector or bonds_TS must be specified.")

    return atoms_IS_new, atoms_FS_new

# -------------------------------------------------------------------------------------
# GET ACTIVATION ENERGIES
# -------------------------------------------------------------------------------------

def get_activation_energies(images):
    """Calculate activation energies from images."""
    energy_TS = max([atoms.get_potential_energy() for atoms in images])
    eact_for = energy_TS-images[0].get_potential_energy()
    eact_rev = energy_TS-images[-1].get_potential_energy()
    return eact_for, eact_rev

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
