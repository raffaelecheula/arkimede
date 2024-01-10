# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.neb import NEB, idpp_interpolate
from ase.io import Trajectory
from ase.optimize import BFGS

# -----------------------------------------------------------------------------
# GET ATOMS FIXED
# -----------------------------------------------------------------------------

def get_atoms_fixed(atoms, return_mask=False):
    
    from ase.constraints import FixAtoms
    indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices += list(constraint.get_indices())
    
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -----------------------------------------------------------------------------
# GET ATOMS NOT FIXED
# -----------------------------------------------------------------------------

def get_atoms_not_fixed(atoms, return_mask=False):
    
    indices = [ii for ii in range(len(atoms)) if ii not in get_atoms_fixed(atoms)]
    
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -----------------------------------------------------------------------------
# GET IDPP INTERPOLATED IMAGES 
# -----------------------------------------------------------------------------

def get_idpp_interpolated_images(
    atoms_IS,
    atoms_FS,
    n_images=10,
    name=None,
    fmax_idpp=0.001,
):
    """Get idpp interpolated images with strict optimization parameters."""
    
    images = [atoms_IS.copy()]
    images += [atoms_IS.copy() for i in range(n_images-2)]
    images += [atoms_FS.copy()]
    neb = NEB(images = images)
    neb.interpolate()
    idpp_interpolate(
        images = images,
        traj = f'{name}_idpp_opt.traj' if name else None,
        log = f'{name}_idpp_opt.log' if name else None, 
        fmax = fmax_idpp,
        optimizer = BFGS,
        mic = False,
        steps = 10000,
    )
    if name:
        traj = Trajectory(filename = f'{name}_idpp.traj', mode = 'w')
        for image in images:
            traj.write(image)

    return images

# -----------------------------------------------------------------------------
# GET ATOMS TS FROM IMAGES NEB
# -----------------------------------------------------------------------------

def get_atoms_TS_from_images_neb(images, atoms_TS=None, index_TS=None):
    """Get the atoms of the TS guess and the vector for dimer calculation 
    from the NEB images."""

    if index_TS is None:
        index_TS = np.argmax([image.get_potential_energy() for image in images])

    if atoms_TS is None:
        atoms_TS = images[index_TS].copy()
    else:
        atoms_TS.set_positions(images[index_TS].positions)
        atoms_TS.info["converged"] = images[index_TS].info["converged"]
    
    vector = np.zeros((len(atoms_TS), 3))
    if index_TS > 0:
        vector += atoms_TS.positions-images[index_TS-1].positions
    if index_TS < len(images)-1:
        vector += images[index_TS+1].positions-atoms_TS.positions
    vector /= np.linalg.norm(vector)
    atoms_TS.info["vector"] = vector
    
    return atoms_TS

# -----------------------------------------------------------------------------
# CHECK NEW IS AND FS
# -----------------------------------------------------------------------------

def get_new_IS_and_FS(
    calc,
    atoms_TS=None,
    images=None,
    vector=None,
    bonds_TS=None,
    delta=0.01,
    sign_bond_dict={'break': +1, 'form': -1},
):

    if images is not None:
        index_TS = np.argmax([image.get_potential_energy() for image in images])
        index_IS = index_TS-1 if index_TS > 0 else index_TS
        index_FS = index_TS+1 if index_TS < len(images) else index_TS
        atoms_IS_new = images[index_IS].copy()
        atoms_FS_new = images[index_FS].copy()
    elif atoms_TS is not None:
        atoms_IS_new = atoms_TS.copy()
        atoms_FS_new = atoms_TS.copy()
        signs_images = [-1, +1]
        if vector is not None:
            for ii, atoms in enumerate([atoms_IS_new, atoms_FS_new]):
                atoms.positions += signs_images[ii] * vector * delta
        if bonds_TS is not None:
            for bond in bonds_TS:
                index_a, index_b, sign_bond = bond
                if isinstance(sign_bond, str):
                    sign_bond = sign_bond_dict[sign_bond]
                dir_bond = atoms_TS.positions[index_a]-atoms_TS.positions[index_b]
                for ii, atoms in enumerate([atoms_IS_new, atoms_FS_new]):
                    sign_tot = sign_bond * signs_images[ii]
                    atoms.positions[index_a] += sign_tot * dir_bond * delta
                    atoms.positions[index_b] -= sign_tot * dir_bond * delta

    return atoms_IS_new, atoms_FS_new

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
