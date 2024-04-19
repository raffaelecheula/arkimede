# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
from ase import Atoms
from ase.neb import NEB, idpp_interpolate
from ase.io import Trajectory
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator, all_properties

# -------------------------------------------------------------------------------------
# GET ATOMS FIXED
# -------------------------------------------------------------------------------------

def get_atoms_fixed(atoms, return_mask=False):
    """Get atoms with FixAtoms constraints."""
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
# GET ATOMS NOT FIXED
# -------------------------------------------------------------------------------------

def get_atoms_not_fixed(atoms, return_mask=False):
    """Get atoms without FixAtoms constraints."""
    indices = [ii for ii in range(len(atoms)) if ii not in get_atoms_fixed(atoms)]
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET ATOMS NOT SURFACE
# -------------------------------------------------------------------------------------

def get_atoms_not_surface(atoms, return_mask=False):
    """Get atoms with index higher than n_atoms_clean."""
    indices = list(range(len(atoms)))[atoms.info["n_atoms_clean"]:]
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -------------------------------------------------------------------------------------
# GET IDPP INTERPOLATED IMAGES 
# -------------------------------------------------------------------------------------

def get_idpp_interpolated_images(
    atoms_IS,
    atoms_FS,
    n_images=10,
    label='idpp',
    fmax=0.005,
    steps_max=1000,
    save_trajs=False,
):
    """Get idpp interpolated images with strict optimization parameters."""
    
    images = [atoms_IS.copy()]
    images += [atoms_IS.copy() for i in range(n_images-2)]
    images += [atoms_FS.copy()]
    neb = NEB(images = images)
    neb.interpolate()
    idpp_interpolate(
        images=images,
        traj=None,
        log=None, 
        fmax=fmax,
        optimizer=BFGS,
        mic=False,
        steps=steps_max,
    )
    if save_trajs is True:
        traj = Trajectory(filename = f'{label}.traj', mode = 'w')
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
# WRITE ATOMS TO DATABASE
# -------------------------------------------------------------------------------------

def write_atoms_to_db(atoms, db_ase):
    """Write atoms to ase database."""
    name = atoms.info["name"]
    calculation = atoms.info["calculation"]
    status = get_status_calculation(atoms)
    species = atoms.info["species"]
    surf_structure = atoms.info["surf_structure"]
    if db_ase.count(name=name, calculation=calculation) == 0:
        db_ase.write(
            atoms=Atoms(atoms),
            name=name,
            calculation=calculation,
            status=status,
            species=species,
            surf_structure=surf_structure,
            data=atoms.info,
        )
    elif db_ase.count(name=name, calculation=calculation) == 1:
        row_id = db_ase.get(name=name, calculation=calculation).id
        db_ase.update(
            id=row_id,
            atoms=Atoms(atoms),
            calculation=calculation,
            status=status,
            species=species,
            surf_structure=surf_structure,
            data=atoms.info,
        )

# -------------------------------------------------------------------------------------
# READ ATOMS FROM DATABASE
# -------------------------------------------------------------------------------------

def read_atoms_from_db(atoms, db_ase):
    """Read atoms from ase database."""
    name = atoms.info["name"]
    calculation = atoms.info["calculation"]
    if db_ase.count(name=name, calculation=calculation) == 1:
        atoms_row = db_ase.get(name=name, calculation=calculation)
        info = atoms_row.data.copy()
        info.update(atoms_row.key_value_pairs)
        atoms.info.update(info)
        atoms.set_positions(atoms_row.positions)
        atoms.calc = SinglePointCalculator(
            atoms=atoms,
            energy=atoms_row.energy,
            forces=atoms_row.forces,
        )
        return atoms
    else:
        return None

# -------------------------------------------------------------------------------------
# UPDATE CLEAN SLAB POSITIONS
# -------------------------------------------------------------------------------------

def update_clean_slab_positions(atoms, db_ase):
    """Update positions of relaxed clean slab."""
    if "atoms_clean" in atoms.info:
        if db_ase.count(name = atoms.info["atoms_clean"]) == 1:
            atoms_row = db_ase.get(name = atoms.info["atoms_clean"])
            atoms.set_positions(
                np.vstack((atoms_row.positions, atoms[len(atoms_row):].positions))
            )
    return atoms

# -------------------------------------------------------------------------------------
# PRINT RESULTS CALCULATION
# -------------------------------------------------------------------------------------

def get_status_calculation(atoms):
    """Get the status of the calculation."""
    if atoms.info["converged"] is True:
        if atoms.info["modified"] is True:
            status = "modified"
        else:
            status = "finished"
    else:
        status = "unfinished"
    return status

# -------------------------------------------------------------------------------------
# PRINT RESULTS CALCULATION
# -------------------------------------------------------------------------------------

def print_results_calculation(atoms):
    """Print results of the calculation to screen."""
    status_str = f'{get_status_calculation(atoms):15s}'
    if atoms.calc and "energy" in atoms.calc.results:
        energy_str = f'{atoms.calc.results["energy"]:15.3f}'
    else:
        energy_str = None
    print(
        f'{atoms.info["name"]:100s}',
        f'{atoms.info["calculation"]:20s}',
        status_str,
        energy_str,
    )

# -------------------------------------------------------------------------------------
# UPDATE INFO TS
# -------------------------------------------------------------------------------------

def update_info_TS(atoms_TS, atoms_IS, atoms_FS):
    """Update positions of relaxed clean slab."""
    species = "→".join([atoms_IS.info["species"], atoms_FS.info["species"]])
    site_id = "→".join([atoms_IS.info["site_id"], atoms_FS.info["site_id"]])
    name_TS = "_".join([atoms_IS.info["surface_name"], species, site_id])
    atoms_TS.info["species"] = species
    atoms_TS.info["site_id"] = site_id
    atoms_TS.info["name"] = name_TS

# -------------------------------------------------------------------------------------
# CHECK SAME CONNECTIVITY
# -------------------------------------------------------------------------------------

def check_same_connectivity(atoms_1, atoms_2):
    """Check if two structures have the same connectivity."""
    from arkimede.catkit.utils.connectivity import get_connectivity
    connectivity_1 = get_connectivity(atoms_1)
    connectivity_2 = get_connectivity(atoms_2)
    return (connectivity_1 == connectivity_2).all()

# -------------------------------------------------------------------------------------
# GET NAMES METADATA
# -------------------------------------------------------------------------------------

def get_names_metadata(
    atoms_gas_tot,
    atoms_clean_tot,
    atoms_ads_tot,
    atoms_neb_tot,
):
    """Get the structures names as dictionary."""
    metadata = {
        "gas": [
            atoms.info["name"] for atoms in atoms_gas_tot
        ],
        "clean": [
            atoms.info["name"] for atoms in atoms_clean_tot
        ],
        "adsorbates": [
            atoms.info["name"] for atoms in atoms_ads_tot
        ],
        "reactions": [
            [atoms.info["name"] for atoms in atoms_neb] 
            for atoms_neb in atoms_neb_tot
        ],
    }
    return metadata

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db(db_ase, selection, structure_type):
    
    atoms_list = []
    for name in db_ase.metadata[structure_type]:
        if isinstance(name, str):
            selection_new = ",".join([selection, f"name={name}"])
            atoms_row = db_ase.get(selection=selection_new)
            atoms = atoms_row.toatoms()
            atoms.info = atoms_row.data
            atoms_list.append(atoms)
        else:
            atoms_ii_list = []
            for name_ii in name:
                selection_new = ",".join([selection, f"name={name_ii}"])
                atoms_row = db_ase.get(selection=selection_new)
                atoms = atoms_row.toatoms()
                atoms.info = atoms_row.data
                atoms_ii_list.append(atoms)
            atoms_list.append(atoms_ii_list)
    
    return atoms_list

# -------------------------------------------------------------------------------------
# FILTER RESULTS
# -------------------------------------------------------------------------------------

def filter_results(results, properties=all_properties):
    return {pp: results[pp] for pp in results if pp in properties}

# -------------------------------------------------------------------------------------
# GET ATOMS TOO CLOSE
# -------------------------------------------------------------------------------------

def get_atoms_too_close(atoms, mindist=0.2, return_anomaly=False):
    """Get indices of atoms too close to each other."""
    indices = []
    for ii, position in enumerate(atoms.positions):
        distances = np.linalg.norm(atoms.positions[ii+1:]-position, axis=1)
        duplicates = np.where(distances < mindist)[0]
        if len(duplicates) > 0:
            indices += [jj+ii+1 for jj in duplicates if jj+ii+1 not in indices]
            if return_anomaly is True:
                break
    
    if return_anomaly is True:
        return len(indices) > 0
    else:
        return indices

# -------------------------------------------------------------------------------------
# READ STEP ACTLEARN
# -------------------------------------------------------------------------------------

def read_step_actlearn(filename):
    """Get the current step of the active learning loop."""
    if os.path.isfile(filename):
        with open(filename, "r") as fileobj:
            step = yaml.safe_load(fileobj)["step_actlearn"]
    else:
        step = 0
        write_step_actlearn(filename=filename, step=step)
    return step

# -------------------------------------------------------------------------------------
# WRITE STEP ACTLEARN
# -------------------------------------------------------------------------------------

def write_step_actlearn(filename, step):
    """Write the new step of the active learning loop."""
    with open(filename, "w") as fileobj:
        yaml.dump({"step_actlearn": step}, fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
