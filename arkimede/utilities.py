# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase import Atoms
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator, all_properties

# -------------------------------------------------------------------------------------
# GET INDICES FIXED
# -------------------------------------------------------------------------------------

def get_indices_fixed(
    atoms: Atoms,
    return_mask: bool = False,
):
    """
    Get indices of atoms with FixAtoms constraints.
    """
    from ase.constraints import FixAtoms
    # Get indices.
    indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices += list(constraint.get_indices())
    # Return indices or mask.
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET INDICES NOT FIXED
# -------------------------------------------------------------------------------------

def get_indices_not_fixed(
    atoms: Atoms,
    return_mask: bool = False,
):
    """
    Get indices of atoms without FixAtoms constraints.
    """
    # Get indices.
    indices = [ii for ii in range(len(atoms)) if ii not in get_indices_fixed(atoms)]
    # Return indices or mask.
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET INDICES ADSORBATE
# -------------------------------------------------------------------------------------

def get_indices_adsorbate(
    atoms: Atoms,
    return_mask: bool = False,
):
    """
    Get indices of atoms adsorbate.
    """
    # Get indices.
    if "indices_ads" in atoms.info:
        indices = atoms.info["indices_ads"]
    else:
        raise RuntimeError("Cannot calculate atoms not surface.")
    # Return indices or mask.
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# FILTER RESULTS
# -------------------------------------------------------------------------------------

def filter_results(
    results: dict,
    properties: list = all_properties,
):
    """
    Filter results to only properties managed by ASE.
    """
    # Return filtered results.
    return {pp: results[pp] for pp in results if pp in properties}

# -------------------------------------------------------------------------------------
# FILTER CONSTRAINTS
# -------------------------------------------------------------------------------------

def filter_constraints(
    atoms: Atoms,
):
    """
    Filter constraints to only constraints managed by ASE.
    """
    from ase.constraints import __all__
    # Remove non-ASE constraints.
    for ii, constraint in reversed(list(enumerate(atoms.constraints))):
        if constraint.todict()["name"] not in __all__:
            del atoms.constraints[ii]
    # Reset atoms in calculator.
    if atoms.calc is not None:
        atoms.calc.atoms = atoms

# -------------------------------------------------------------------------------------
# UPDATE ATOMS FROM ATOMS OPT
# -------------------------------------------------------------------------------------

def update_atoms_from_atoms_opt(
    atoms: Atoms,
    atoms_opt: Atoms,
    status: str,
    properties: list = all_properties,
    update_cell: bool = False,
):
    """
    Update the input atoms with the results of the optimized atoms.
    """
    # Update positions.
    atoms.set_positions(atoms_opt.get_positions())
    # Update cell.
    if update_cell is True:
        atoms.set_cell(atoms_opt.get_cell())
    # Update calculator results.
    results = filter_results(results=atoms_opt.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    # Update info.
    atoms.info.update(atoms_opt.info)
    atoms.info["status"] = status
    if "counter" in dir(atoms_opt.calc):
        atoms.info["counter"] = atoms_opt.calc.counter
    # Return atoms.
    return atoms

# -------------------------------------------------------------------------------------
# COPY ATOMS AND RESULTS
# -------------------------------------------------------------------------------------

def copy_atoms_and_results(
    atoms: Atoms,
    new_calc: bool = True,
    properties: list = all_properties,
):
    """
    Copy an ase Atoms object along with its calculator results.
    """
    # Copy atoms.
    atoms_copy = atoms.copy()
    # Set calculator.
    if new_calc is True:
        atoms_copy.calc = SinglePointCalculator(atoms=atoms_copy)
    else:
        atoms_copy.calc = deepcopy(atoms.calc)
    # Filter results.
    atoms_copy.calc.results = filter_results(
        results=atoms.calc.results,
        properties=properties,
    )
    # Filter constraints.
    filter_constraints(atoms=atoms_copy)
    atoms_copy.calc.atoms = atoms_copy
    # Return atoms copy.
    return atoms_copy

# -------------------------------------------------------------------------------------
# GET EDGES LIST
# -------------------------------------------------------------------------------------

def get_edges_list(
    atoms: Atoms,
    indices: list = None,
    dist_ratio_thr: float = 1.25,
):
    """
    Get the edges for selected atoms in an ase Atoms object.
    """
    from itertools import combinations
    from ase.neighborlist import natural_cutoffs
    # Get indices.
    if indices is None:
        indices = range(len(atoms))
    indices = [int(ii) for ii in indices]
    # Get edges list.
    edges_list = []
    cutoffs = natural_cutoffs(atoms=atoms)
    for combo in combinations(list(indices), 2):
        total_distance = atoms.get_distance(combo[0], combo[1], mic=True)
        r1 = cutoffs[combo[0]]
        r2 = cutoffs[combo[1]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio <= dist_ratio_thr:
            edges_list.append([int(ii) for ii in combo])
    # Return edges list.
    return edges_list

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY
# -------------------------------------------------------------------------------------

def get_connectivity(
    atoms: Atoms,
    edges_list: list = None,
    indices: list = None,
    dist_ratio_thr: float = 1.25,
):
    """
    Get the connectivity matrix for selected atoms in an ase Atoms object.
    """
    # Get edges list.
    if edges_list is None:
        edges_list = get_edges_list(
            atoms=atoms,
            indices=indices,
            dist_ratio_thr=dist_ratio_thr,
        )
    # Populate connectivity matrix.
    matrix = np.zeros((len(atoms), len(atoms)))
    for aa, bb in edges_list:
        matrix[aa, bb] += 1
        matrix[bb, aa] += 1
    # Return connectivity matrix.
    return matrix

# -------------------------------------------------------------------------------------
# CHECK SAME CONNECTIVITY
# -------------------------------------------------------------------------------------

def check_same_connectivity(
    atoms_1: Atoms,
    atoms_2: Atoms,
    indices: list = None,
):
    """
    Check if two structures have the same connectivity.
    """
    # Get indices.
    if indices == "adsorbate":
        indices = get_indices_adsorbate(atoms=atoms_1)
    elif indices == "not_fixed":
        indices = get_indices_not_fixed(atoms=atoms_1)
    # Get connectivity matrices.
    connectivity_1 = get_connectivity(atoms=atoms_1, indices=indices)
    connectivity_2 = get_connectivity(atoms=atoms_2, indices=indices)
    # Return comparison.
    return bool((connectivity_1 == connectivity_2).all())

# -------------------------------------------------------------------------------------
# PRINT TITLE
# -------------------------------------------------------------------------------------

def print_title(
    string: str,
    width: int = 100,
):
    """
    Print title.
    """
    for text in ["-" * width, string.center(width), "-" * width]:
        print("#", text, "#")

# -------------------------------------------------------------------------------------
# PRINT SUBTITLE
# -------------------------------------------------------------------------------------

def print_subtitle(
    string: str,
    width: int = 100,
    newline_before: bool = True,
    newline_after: bool = False,
):
    """
    Print subtitle.
    """
    if newline_before is True:
        print("")
    print("#", f" {string} ".center(width, "-"), "#")
    if newline_after is True:
        print("")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
