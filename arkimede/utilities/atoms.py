# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------------------------
# GET INDICES FIXED
# -------------------------------------------------------------------------------------

def get_indices_fixed(atoms, return_mask=False):
    """Get indices of atoms with FixAtoms constraints."""
    from ase.constraints import FixAtoms
    indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices += list(constraint.get_indices())
    
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET INDICES NOT FIXED
# -------------------------------------------------------------------------------------

def get_indices_not_fixed(atoms, return_mask=False):
    """Get indices of atoms without FixAtoms constraints."""
    indices = [ii for ii in range(len(atoms)) if ii not in get_indices_fixed(atoms)]
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET INDICES ADSORBATE
# -------------------------------------------------------------------------------------

def get_indices_adsorbate(atoms, return_mask=False):
    """Get indices of atoms adsorbate."""
    if "indices_ads" in atoms.info:
        indices = atoms.info["indices_ads"]
    elif "n_atoms_clean" in atoms.info: # TODO: remove this option?
        indices = list(range(len(atoms)))[atoms.info["n_atoms_clean"]:]
    else:
        raise RuntimeError("Cannot calculate atoms not surface.")
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET EDGES LIST
# -------------------------------------------------------------------------------------

def get_edges_list(atoms, indices=None, dist_ratio_thr=1.25):
    """Get the edges for selected atoms in an ase Atoms object."""
    from itertools import combinations
    from ase.neighborlist import natural_cutoffs
    if indices is None:
        indices = range(len(atoms))
    indices = [int(ii) for ii in indices]
    edges_list = []
    cutoffs = natural_cutoffs(atoms=atoms)
    for combo in combinations(list(indices), 2):
        total_distance = atoms.get_distance(combo[0], combo[1], mic=True)
        r1 = cutoffs[combo[0]]
        r2 = cutoffs[combo[1]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio <= dist_ratio_thr:
            edges_list.append([int(ii) for ii in combo])
    return edges_list

# -------------------------------------------------------------------------------------
# GET CONNECTVITY
# -------------------------------------------------------------------------------------

def get_connectivity(atoms, edges_list=None, indices=None, dist_ratio_thr=1.25):
    """Get the connectivity matrix for selected atoms in an ase Atoms object."""
    if edges_list is None:
        edges_list = get_edges_list(
            atoms=atoms,
            indices=indices,
            dist_ratio_thr=dist_ratio_thr,
        )
    matrix = np.zeros((len(atoms), len(atoms)))
    for aa, bb in edges_list:
        matrix[aa, bb] += 1
        matrix[bb, aa] += 1
    return matrix

# -------------------------------------------------------------------------------------
# UPDATE CLEAN SLAB POSITIONS
# -------------------------------------------------------------------------------------

def update_clean_slab_positions(atoms, db_ase):
    """Update positions of relaxed clean slab."""
    # TODO: use indices_ads instead.
    if "name_ref" in atoms.info and "n_atoms_clean" in atoms.info:
        if db_ase.count(name = atoms.info["name_ref"]) == 1:
            atoms_row = db_ase.get(name = atoms.info["name_ref"])
            n_atoms_clean = atoms_row.data["n_atoms_clean"]
            atoms.set_positions(
                np.vstack((atoms_row.positions, atoms[n_atoms_clean:].positions))
            )
    return atoms

# -------------------------------------------------------------------------------------
# CHECK SAME CONNECTIVITY
# -------------------------------------------------------------------------------------

def check_same_connectivity(atoms_1, atoms_2, indices=None):
    """Check if two structures have the same connectivity."""
    #from arkimede.catkit.utils.connectivity import get_connectivity
    if indices is None:
        indices = atoms_1.info["indices_ads"]
    connectivity_1 = get_connectivity(atoms_1)
    connectivity_2 = get_connectivity(atoms_2)
    return (connectivity_1 == connectivity_2).all()

# -------------------------------------------------------------------------------------
# GET ATOMS MIN ENERGY
# -------------------------------------------------------------------------------------

def get_atoms_min_energy(atoms_list):
    ii = np.argmin([atoms.get_potential_energy() for atoms in atoms_list])
    return atoms_list[ii]

# -------------------------------------------------------------------------------------
# GET ATOMS MAX ENERGY
# -------------------------------------------------------------------------------------

def get_atoms_max_energy(atoms_list):
    ii = np.argmax([atoms.get_potential_energy() for atoms in atoms_list])
    return atoms_list[ii]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
